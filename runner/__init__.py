"""
Compatibility shims for legacy mmcv.runner API using mmengine.
Only the minimal subset used by this repo is implemented.
"""
from typing import Any, Dict, Optional, List, Tuple

import torch

from mmengine.runner import load_checkpoint as _mmengine_load_checkpoint
from mmengine.dist import get_dist_info as _mmengine_get_dist_info
from mmengine.hooks import DistSamplerSeedHook, Hook  # noqa: F401


def load_checkpoint(model: torch.nn.Module, filename: str, *args, **kwargs):
    """Alias to mmengine.runner.load_checkpoint."""
    return _mmengine_load_checkpoint(model, filename, *args, **kwargs)


def get_dist_info(*args, **kwargs):
    """Alias to mmengine.dist.get_dist_info."""
    return _mmengine_get_dist_info(*args, **kwargs)


def obj_from_dict(info: Dict[str, Any], module, default_args: Optional[Dict] = None):
    """Build an object from a config dict and a python module.

    This mimics the old mmcv.runner.obj_from_dict used in this codebase.
    The dict must contain key 'type' that is either a class object or a
    string name of a class under the given module.
    """
    assert isinstance(info, dict) and 'type' in info
    args = info.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        if not hasattr(module, obj_type):
            raise KeyError(f'{obj_type} is not found in module {module!r}')
        obj_type = getattr(module, obj_type)
    elif not isinstance(obj_type, type):
        raise TypeError(f'type must be str or type, but got {type(obj_type)}')
    if default_args is not None:
        for k, v in default_args.items():
            args.setdefault(k, v)
    return obj_type(**args)


class OptimizerHook:
    """Minimal base hook for optimizer steps used by DistOptimizerHook.

    Provides gradient clipping utilities compatible with expected usage.
    """

    def __init__(self, grad_clip: Optional[Dict] = None):
        self.grad_clip = grad_clip

    @staticmethod
    def clip_grads(params, max_norm=35.0, norm_type=2.0):
        torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    # Placeholder for hook interface
    def after_train_iter(self, runner):  # pragma: no cover - not used here
        raise NotImplementedError


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict) -> torch.optim.Optimizer:
    """Build a PyTorch optimizer from config.

    Args:
        model: The model whose parameters will be optimized.
        optim_cfg: Dict like dict(type='SGD', lr=..., momentum=..., ...).
    """
    cfg = optim_cfg.copy()
    optim_type = cfg.pop('type')
    optimizer_cls = getattr(torch.optim, optim_type)
    return optimizer_cls(model.parameters(), **cfg)


class LogBuffer:
    def __init__(self):
        self.output: Dict[str, Any] = {}
        self.ready: bool = False


class EpochBasedRunner:  # minimal training loop to satisfy legacy API
    def __init__(self, model: torch.nn.Module, batch_processor, optimizer,
                 work_dir: str, logger):
        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.epoch = 0
        self.iter = 0
        self.mode = 'train'
        self._hooks: List[Any] = []
        self.log_buffer = LogBuffer()
        self.rank, self.world_size = _mmengine_get_dist_info()

        # training hook placeholders
        self._optimizer_hook: Optional[OptimizerHook] = None

    def register_training_hooks(self, lr_config, optimizer_config,
                                checkpoint_config, log_config):
        # Only optimizer hook is respected in this minimal runner
        self._optimizer_hook = optimizer_config

    def register_hook(self, hook):
        self._hooks.append(hook)

    def resume(self, checkpoint: str):  # pragma: no cover
        _mmengine_load_checkpoint(self.model, checkpoint)

    def load_checkpoint(self, checkpoint: str):  # pragma: no cover
        _mmengine_load_checkpoint(self.model, checkpoint)

    def call_hook(self, fn_name: str):
        for hook in self._hooks:
            fn = getattr(hook, fn_name, None)
            if callable(fn):
                fn(self)

    def run(self, data_loaders: List[Any], workflow: List[Tuple[str, int]],
            total_epochs: int):
        assert len(data_loaders) == 1, 'This minimal runner supports one dataloader'
        data_loader = data_loaders[0]
        max_epochs = total_epochs
        self.model.train()
        for epoch in range(max_epochs):
            self.epoch = epoch
            # before epoch hooks
            self.call_hook('before_train_epoch')

            for i, data in enumerate(data_loader):
                self.iter += 1
                outputs = self.batch_processor(self.model, data, train_mode=True)
                self.outputs = outputs  # for hooks
                if self._optimizer_hook is not None:
                    # Delegate step to optimizer hook if provided
                    self._optimizer_hook.after_train_iter(self)
                else:
                    self.optimizer.zero_grad()
                    outputs['loss'].backward()
                    self.optimizer.step()

                # simple logging
                self.log_buffer.output.update(outputs.get('log_vars', {}))
                self.log_buffer.ready = True

            # after epoch hooks
            self.call_hook('after_train_epoch')


def parallel_test(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError('mmcv.runner.parallel_test is not implemented in this environment.')
