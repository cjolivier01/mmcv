# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import warnings
from typing import List, Optional, Tuple, Union, no_type_check

import cv2
import numpy as np
from mmengine.utils import to_2tuple

# Optional torch/kornia support for GPU tensors
try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    F = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import kornia as K  # noqa: F401
    from kornia.geometry.transform import (
        get_rotation_matrix2d as kornia_get_rotation_matrix2d,)
    from kornia.geometry.transform import warp_affine as kornia_warp_affine
except Exception:  # pragma: no cover - kornia optional
    kornia_warp_affine = None  # type: ignore
    kornia_get_rotation_matrix2d = None  # type: ignore

from .io import imread_backend

try:
    from PIL import Image
except ImportError:
    Image = None


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

cv2_border_modes = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
    'reflect_101': cv2.BORDER_REFLECT_101,
    'transparent': cv2.BORDER_TRANSPARENT,
    'isolated': cv2.BORDER_ISOLATED
}

# Pillow >=v9.1.0 use a slightly different naming scheme for filters.
# Set pillow_interp_codes according to the naming scheme used.
if Image is not None:
    if hasattr(Image, 'Resampling'):
        pillow_interp_codes = {
            'nearest': Image.Resampling.NEAREST,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'box': Image.Resampling.BOX,
            'lanczos': Image.Resampling.LANCZOS,
            'hamming': Image.Resampling.HAMMING
        }
    else:
        pillow_interp_codes = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }


def imresize(
    img: Union[np.ndarray, 'torch.Tensor'],
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None
) -> Union[Tuple[Union[np.ndarray, 'torch.Tensor'], float, float], np.ndarray, 'torch.Tensor']:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    # Torch path (no numpy conversion allowed)
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if backend == 'pillow':
            raise ValueError('Pillow backend is not supported for torch.Tensor inputs')

        # Determine layout and convert to BCHW
        orig = img
        layout = None
        if img.dim() == 2:  # H, W
            h, w = int(img.shape[0]), int(img.shape[1])
            img_bchw = img.unsqueeze(0).unsqueeze(0)
            layout = 'HW'
        elif img.dim() == 3:
            # Heuristic: treat as HWC if last dim is small (<=4); else CHW
            if img.shape[-1] <= 4:
                h, w = int(img.shape[0]), int(img.shape[1])
                img_bchw = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
                layout = 'HWC'
            else:
                # CHW
                h, w = int(img.shape[1]), int(img.shape[2])
                img_bchw = img.unsqueeze(0)
                layout = 'CHW'
        elif img.dim() == 4:
            # Assume BCHW if channel dim is 1/3/4 else BHWC
            if img.shape[1] in (1, 3, 4):
                h, w = int(img.shape[2]), int(img.shape[3])
                img_bchw = img
                layout = 'BCHW'
            else:
                h, w = int(img.shape[1]), int(img.shape[2])
                img_bchw = img.permute(0, 3, 1, 2)
                layout = 'BHWC'
        else:
            raise ValueError(f'Unsupported torch image rank {img.dim()}')

        # Map interpolation to torch interpolate modes
        interp_map = {
            'nearest': 'nearest',
            'bilinear': 'bilinear',
            'bicubic': 'bicubic',
            'area': 'area',
            'lanczos': 'bicubic',  # best-effort fallback
        }
        if interpolation not in interp_map:
            raise ValueError(f'Unsupported interpolation for torch: {interpolation}')
        mode = interp_map[interpolation]
        if interpolation == 'lanczos':
            warnings.warn('lanczos not supported in torch interpolate; using bicubic instead')

        target_h, target_w = int(size[1]), int(size[0])
        # F.interpolate expects floating for non-nearest; keep dtype if already float
        orig_dtype = img_bchw.dtype
        need_float = mode in ('bilinear', 'bicubic', 'area') and not torch.is_floating_point(img_bchw)
        if need_float:
            img_bchw = img_bchw.to(torch.float32)

        # Build kwargs; align_corners only for linear modes; antialias for bilinear/bicubic
        kwargs = dict(size=(target_h, target_w), mode=mode)
        if mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = False  # type: ignore[index]
            kwargs['antialias'] = True  # type: ignore[index]
        resized = F.interpolate(img_bchw, **kwargs)

        # Cast back to original dtype if needed
        if need_float and not torch.is_floating_point(orig):
            # mimic numpy rounding behavior
            resized = resized.round().to(orig.dtype)

        # Restore original layout
        if layout == 'HW':
            resized_img = resized.squeeze(0).squeeze(0)
        elif layout == 'HWC':
            resized_img = resized.squeeze(0).permute(1, 2, 0)
        elif layout == 'CHW':
            resized_img = resized.squeeze(0)
        elif layout == 'BCHW':
            resized_img = resized
        elif layout == 'BHWC':
            resized_img = resized.permute(0, 2, 3, 1)
        else:
            raise AssertionError('Unexpected layout during resize')

        if not return_scale:
            return resized_img  # type: ignore[return-value]
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale  # type: ignore[return-value]

    # Numpy/OpenCV path
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


@no_type_check
def imresize_to_multiple(
    img: Union[np.ndarray, 'torch.Tensor'],
    divisor: Union[int, Tuple[int, int]],
    size: Union[int, Tuple[int, int], None] = None,
    scale_factor: Union[float, int, Tuple[float, float], Tuple[int, int],
                        None] = None,
    keep_ratio: bool = False,
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None
) -> Union[Tuple[Union[np.ndarray, 'torch.Tensor'], float, float], np.ndarray, 'torch.Tensor']:
    """Resize image according to a given size or scale factor and then rounds
    up the the resized or rescaled image size to the nearest value that can be
    divided by the divisor.

    Args:
        img (ndarray): The input image.
        divisor (int | tuple): Resized image size will be a multiple of
            divisor. If divisor is a tuple, divisor should be
            (w_divisor, h_divisor).
        size (None | int | tuple[int]): Target size (w, h). Default: None.
        scale_factor (None | float | int | tuple[float] | tuple[int]):
            Multiplier for spatial size. Should match input size if it is a
            tuple and the 2D style is (w_scale_factor, h_scale_factor).
            Default: None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: False.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        # get (h, w) for torch tensors
        if img.dim() == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
        elif img.dim() == 3:
            if img.shape[-1] <= 4:
                h, w = int(img.shape[0]), int(img.shape[1])
            else:
                h, w = int(img.shape[1]), int(img.shape[2])
        elif img.dim() == 4:
            if img.shape[1] in (1, 3, 4):
                h, w = int(img.shape[2]), int(img.shape[3])
            else:
                h, w = int(img.shape[1]), int(img.shape[2])
        else:
            raise ValueError(f'Unsupported torch image rank {img.dim()}')
    else:
        h, w = img.shape[:2]
    if size is not None and scale_factor is not None:
        raise ValueError('only one of size or scale_factor should be defined')
    elif size is None and scale_factor is None:
        raise ValueError('one of size or scale_factor should be defined')
    elif size is not None:
        size = to_2tuple(size)
        if keep_ratio:
            size = rescale_size((w, h), size, return_scale=False)
    else:
        size = _scale_size((w, h), scale_factor)

    divisor = to_2tuple(divisor)
    size = tuple(int(np.ceil(s / d)) * d for s, d in zip(size, divisor))
    resized_img, w_scale, h_scale = imresize(
        img,
        size,
        return_scale=True,
        interpolation=interpolation,
        out=out,
        backend=backend)
    if return_scale:
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def imresize_like(
    img: Union[np.ndarray, 'torch.Tensor'],
    dst_img: Union[np.ndarray, 'torch.Tensor'],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = None
) -> Union[Tuple[Union[np.ndarray, 'torch.Tensor'], float, float], np.ndarray, 'torch.Tensor']:
    """Resize image to the same size of a given image.

    Args:
        img (ndarray): The input image.
        dst_img (ndarray): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    if 'torch' in globals() and torch is not None and isinstance(dst_img, torch.Tensor):
        if dst_img.dim() == 2:
            h, w = int(dst_img.shape[0]), int(dst_img.shape[1])
        elif dst_img.dim() == 3:
            if dst_img.shape[-1] <= 4:
                h, w = int(dst_img.shape[0]), int(dst_img.shape[1])
            else:
                h, w = int(dst_img.shape[1]), int(dst_img.shape[2])
        elif dst_img.dim() == 4:
            if dst_img.shape[1] in (1, 3, 4):
                h, w = int(dst_img.shape[2]), int(dst_img.shape[3])
            else:
                h, w = int(dst_img.shape[1]), int(dst_img.shape[2])
        else:
            raise ValueError(f'Unsupported torch image rank {dst_img.dim()}')
    else:
        h, w = dst_img.shape[:2]
    return imresize(img, (w, h), return_scale, interpolation, backend=backend)


def rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: Union[np.ndarray, 'torch.Tensor'],
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = None
) -> Union[np.ndarray, 'torch.Tensor', Tuple[Union[np.ndarray, 'torch.Tensor'], float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if img.dim() == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
        elif img.dim() == 3:
            if img.shape[-1] <= 4:
                h, w = int(img.shape[0]), int(img.shape[1])
            else:
                h, w = int(img.shape[1]), int(img.shape[2])
        elif img.dim() == 4:
            if img.shape[1] in (1, 3, 4):
                h, w = int(img.shape[2]), int(img.shape[3])
            else:
                h, w = int(img.shape[1]), int(img.shape[2])
        else:
            raise ValueError(f'Unsupported torch image rank {img.dim()}')
    else:
        h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imflip(img: Union[np.ndarray, 'torch.Tensor'], direction: str = 'horizontal') -> Union[np.ndarray, 'torch.Tensor']:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if img.dim() == 2:
            if direction == 'horizontal':
                return torch.flip(img, dims=[1])
            elif direction == 'vertical':
                return torch.flip(img, dims=[0])
            else:
                return torch.flip(img, dims=[0, 1])
        elif img.dim() == 3:
            # CHW or HWC
            if img.shape[-1] <= 4:
                # HWC
                if direction == 'horizontal':
                    return torch.flip(img, dims=[1])
                elif direction == 'vertical':
                    return torch.flip(img, dims=[0])
                else:
                    return torch.flip(img, dims=[0, 1])
            else:
                # CHW
                if direction == 'horizontal':
                    return torch.flip(img, dims=[2])
                elif direction == 'vertical':
                    return torch.flip(img, dims=[1])
                else:
                    return torch.flip(img, dims=[1, 2])
        elif img.dim() == 4:
            # BCHW or BHWC
            if img.shape[1] in (1, 3, 4):
                if direction == 'horizontal':
                    return torch.flip(img, dims=[3])
                elif direction == 'vertical':
                    return torch.flip(img, dims=[2])
                else:
                    return torch.flip(img, dims=[2, 3])
            else:  # BHWC
                if direction == 'horizontal':
                    return torch.flip(img, dims=[2])
                elif direction == 'vertical':
                    return torch.flip(img, dims=[1])
                else:
                    return torch.flip(img, dims=[1, 2])
        else:
            raise ValueError(f'Unsupported torch image rank {img.dim()}')
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def imflip_(img: Union[np.ndarray, 'torch.Tensor'], direction: str = 'horizontal') -> Union[np.ndarray, 'torch.Tensor']:
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        flipped = imflip(img, direction)
        img.copy_(flipped)
        return img
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)


def imrotate(img: Union[np.ndarray, 'torch.Tensor'],
             angle: float,
             center: Optional[Tuple[float, float]] = None,
             scale: float = 1.0,
             border_value: int = 0,
             interpolation: str = 'bilinear',
             auto_bound: bool = False,
             border_mode: str = 'constant') -> Union[np.ndarray, 'torch.Tensor']:
    """Rotate an image.

    Args:
        img (np.ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value used in case of a constant border.
            Defaults to 0.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
        border_mode (str): Pixel extrapolation method. Defaults to 'constant'.

    Returns:
        np.ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    # Torch branch
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if img.dim() == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
            img_bchw = img.unsqueeze(0).unsqueeze(0)
            layout = 'HW'
        elif img.dim() == 3:
            if img.shape[-1] <= 4:  # HWC
                h, w = int(img.shape[0]), int(img.shape[1])
                img_bchw = img.permute(2, 0, 1).unsqueeze(0)
                layout = 'HWC'
            else:  # CHW
                h, w = int(img.shape[1]), int(img.shape[2])
                img_bchw = img.unsqueeze(0)
                layout = 'CHW'
        elif img.dim() == 4:
            if img.shape[1] in (1, 3, 4):  # BCHW
                h, w = int(img.shape[2]), int(img.shape[3])
                img_bchw = img
                layout = 'BCHW'
            else:  # BHWC
                h, w = int(img.shape[1]), int(img.shape[2])
                img_bchw = img.permute(0, 3, 1, 2)
                layout = 'BHWC'
        else:
            raise ValueError(f'Unsupported torch image rank {img.dim()}')

        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        # Compute rotation matrix (degrees, note negative to match OpenCV sign)
        if kornia_get_rotation_matrix2d is None or kornia_warp_affine is None:
            raise RuntimeError('kornia is required for torch-based imrotate. Please install kornia.')
        device = img_bchw.device
        dtype = torch.float32
        M = kornia_get_rotation_matrix2d(
            torch.tensor([center], device=device, dtype=dtype),  # (1,2)
            torch.tensor([-angle], device=device, dtype=dtype),  # (1,)
            torch.tensor([[scale, scale]], device=device, dtype=dtype),  # (1,2)
        )[0].unsqueeze(0)  # (1, 2, 3)

        out_w, out_h = w, h
        if auto_bound:
            cos = float(torch.abs(M[0, 0, 0]).item())
            sin = float(torch.abs(M[0, 0, 1]).item())
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            # adjust translation to center the result
            M[:, 0, 2] += (new_w - w) * 0.5
            M[:, 1, 2] += (new_h - h) * 0.5
            out_w = int(np.round(new_w))
            out_h = int(np.round(new_h))

        # Interp and padding modes
        interp = 'bilinear' if interpolation not in ('nearest', 'bilinear') else interpolation
        pad_mode_map = {
            'constant': 'zeros',
            'replicate': 'border',
            'reflect': 'reflection',
        }
        if border_mode not in pad_mode_map:
            raise ValueError(f'Torch imrotate does not support border_mode {border_mode}')
        padding_mode = pad_mode_map[border_mode]

        # Ensure float for warp then cast back
        orig_dtype = img_bchw.dtype
        need_float = not torch.is_floating_point(img_bchw)
        if need_float:
            img_bchw = img_bchw.to(torch.float32)

        # Try passing fill_value if kornia supports
        try:
            rotated = kornia_warp_affine(
                img_bchw, M, dsize=(out_h, out_w),
                mode=interp, padding_mode=padding_mode,
                align_corners=False, fill_value=border_value,
            )
        except TypeError:
            rotated = kornia_warp_affine(
                img_bchw, M, dsize=(out_h, out_w),
                mode=interp, padding_mode=padding_mode,
                align_corners=False,
            )

        if need_float and not torch.is_floating_point(img):
            rotated = rotated.round().to(img.dtype)

        # Restore layout
        if layout == 'HW':
            out = rotated.squeeze(0).squeeze(0)
        elif layout == 'HWC':
            out = rotated.squeeze(0).permute(1, 2, 0)
        elif layout == 'CHW':
            out = rotated.squeeze(0)
        elif layout == 'BCHW':
            out = rotated
        elif layout == 'BHWC':
            out = rotated.permute(0, 2, 3, 1)
        else:
            raise AssertionError('Unexpected layout during rotate')
        return out  # type: ignore[return-value]

    # Numpy/OpenCV branch
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderMode=cv2_border_modes[border_mode],
        borderValue=border_value)
    return rotated


def bbox_clip(bboxes: Union[np.ndarray, 'torch.Tensor'], img_shape: Tuple[int, int]) -> Union[np.ndarray, 'torch.Tensor']:
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple[int]): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    if 'torch' in globals() and torch is not None and isinstance(bboxes, torch.Tensor):
        cmin = torch.empty(bboxes.shape[-1], dtype=bboxes.dtype, device=bboxes.device)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        return torch.minimum(torch.clamp(bboxes, min=0), cmin)  # type: ignore[return-value]
    else:
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
        return clipped_bboxes


def bbox_scaling(bboxes: Union[np.ndarray, 'torch.Tensor'],
                 scale: float,
                 clip_shape: Optional[Tuple[int, int]] = None) -> Union[np.ndarray, 'torch.Tensor']:
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if 'torch' in globals() and torch is not None and isinstance(bboxes, torch.Tensor):
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.clone()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            delta = torch.stack((-dw, -dh, dw, dh), dim=-1)
            scaled_bboxes = bboxes + delta
        if clip_shape is not None:
            return bbox_clip(scaled_bboxes, clip_shape)
        else:
            return scaled_bboxes
    else:
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.copy()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
        if clip_shape is not None:
            return bbox_clip(scaled_bboxes, clip_shape)
        else:
            return scaled_bboxes


def imcrop(
    img: Union[np.ndarray, 'torch.Tensor'],
    bboxes: Union[np.ndarray, 'torch.Tensor'],
    scale: float = 1.0,
    pad_fill: Union[float, list, None] = None
) -> Union[np.ndarray, List[np.ndarray], 'torch.Tensor', List['torch.Tensor']]:
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no scaling.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.

    Returns:
        list[ndarray] | ndarray: The cropped image patches.
    """
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        # Determine channels
        if img.dim() == 2:
            chn = 1
            H, W = int(img.shape[0]), int(img.shape[1])
        elif img.dim() == 3:
            if img.shape[-1] <= 4:  # HWC
                chn = int(img.shape[-1])
                H, W = int(img.shape[0]), int(img.shape[1])
            else:  # CHW
                chn = int(img.shape[0])
                H, W = int(img.shape[1]), int(img.shape[2])
        else:
            raise ValueError('imcrop only supports 2D/3D tensors')
        if pad_fill is not None:
            if isinstance(pad_fill, (int, float)):
                pad_fill_list = [pad_fill for _ in range(chn)]
            else:
                pad_fill_list = pad_fill
                assert len(pad_fill_list) == chn

        _bboxes = bboxes.unsqueeze(0) if isinstance(bboxes, torch.Tensor) and bboxes.ndim == 1 else bboxes
        if isinstance(_bboxes, torch.Tensor):
            scaled_bboxes = bbox_scaling(_bboxes, scale).to(torch.int32)
        else:
            raise TypeError('Torch imcrop expects torch.Tensor bboxes when img is torch.Tensor')
        clipped_bbox = bbox_clip(scaled_bboxes, (H, W))

        patches: List[torch.Tensor] = []
        for i in range(int(clipped_bbox.shape[0])):
            x1, y1, x2, y2 = tuple(int(v) for v in clipped_bbox[i, :])
            if pad_fill is None:
                if img.dim() == 2:
                    patch = img[y1:y2 + 1, x1:x2 + 1]
                elif img.dim() == 3 and img.shape[-1] <= 4:
                    patch = img[y1:y2 + 1, x1:x2 + 1, :]
                else:  # CHW
                    patch = img[:, y1:y2 + 1, x1:x2 + 1]
            else:
                _x1, _y1, _x2, _y2 = tuple(int(v) for v in scaled_bboxes[i, :])
                patch_h = _y2 - _y1 + 1
                patch_w = _x2 - _x1 + 1
                if img.dim() == 2:
                    patch_shape = (patch_h, patch_w)
                elif img.dim() == 3 and img.shape[-1] <= 4:  # HWC
                    patch_shape = (patch_h, patch_w, chn)
                else:  # CHW
                    patch_shape = (chn, patch_h, patch_w)
                fill_tensor = torch.tensor(pad_fill_list, dtype=img.dtype, device=img.device)
                if img.dim() == 2:
                    patch = torch.full(patch_shape, fill_tensor[0].item(), dtype=img.dtype, device=img.device)
                elif img.dim() == 3 and img.shape[-1] <= 4:  # HWC
                    patch = fill_tensor.view(1, 1, chn).expand(patch_h, patch_w, chn).clone()
                else:  # CHW
                    patch = fill_tensor.view(chn, 1, 1).expand(chn, patch_h, patch_w).clone()
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                if img.dim() == 2:
                    patch[y_start:y_start + h, x_start:x_start + w] = img[y1:y1 + h, x1:x1 + w]
                elif img.dim() == 3 and img.shape[-1] <= 4:
                    patch[y_start:y_start + h, x_start:x_start + w, :] = img[y1:y1 + h, x1:x1 + w, :]
                else:
                    patch[:, y_start:y_start + h, x_start:x_start + w] = img[:, y1:y1 + h, x1:x1 + w]
            patches.append(patch)

        if bboxes.ndim == 1:
            return patches[0]
        else:
            return patches

    # Numpy branch
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            patch_h = _y2 - _y1 + 1
            patch_w = _x2 - _x1 + 1
            if chn == 1:
                patch_shape = (patch_h, patch_w)
            else:
                patch_shape = (patch_h, patch_w, chn)  # type: ignore
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start + w,
                  ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def impad(img: Union[np.ndarray, 'torch.Tensor'],
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> Union[np.ndarray, 'torch.Tensor']:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """
    assert (shape is not None) ^ (padding is not None)

    # Torch branch
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if shape is not None:
            if img.dim() == 2:
                H, W = int(img.shape[0]), int(img.shape[1])
            elif img.dim() == 3:
                if img.shape[-1] <= 4:  # HWC
                    H, W = int(img.shape[0]), int(img.shape[1])
                else:  # CHW
                    H, W = int(img.shape[1]), int(img.shape[2])
            else:
                raise ValueError('Torch impad only supports 2D/3D tensors')
            width = max(shape[1] - W, 0)
            height = max(shape[0] - H, 0)
            padding = (0, 0, width, height)

        # normalize padding tuple (l, t, r, b)
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be an int or a 2/4 element tuple. '
                             f'But received {padding}')

        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        # Build output tensor and copy (for constant) or use F.pad for others
        if padding_mode == 'constant':
            # compute output shape and fill
            if img.dim() == 2:
                H, W = int(img.shape[0]), int(img.shape[1])
                out = torch.full((H + padding[1] + padding[3], W + padding[0] + padding[2]),
                                 float(pad_val if isinstance(pad_val, numbers.Number) else pad_val[0]),
                                 dtype=img.dtype, device=img.device)
                out[padding[1]:padding[1] + H, padding[0]:padding[0] + W] = img
                return out  # type: ignore[return-value]
            elif img.dim() == 3:
                if img.shape[-1] <= 4:  # HWC
                    H, W, C = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                    out = torch.empty((H + padding[1] + padding[3], W + padding[0] + padding[2], C),
                                      dtype=img.dtype, device=img.device)
                    if isinstance(pad_val, numbers.Number):
                        out.fill_(float(pad_val))
                    else:
                        pv = torch.tensor(pad_val, dtype=img.dtype, device=img.device).view(1, 1, C)
                        out[:] = pv
                    out[padding[1]:padding[1] + H, padding[0]:padding[0] + W, :] = img
                    return out  # type: ignore[return-value]
                else:  # CHW
                    C, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                    out = torch.empty((C, H + padding[1] + padding[3], W + padding[0] + padding[2]),
                                      dtype=img.dtype, device=img.device)
                    if isinstance(pad_val, numbers.Number):
                        out.fill_(float(pad_val))
                    else:
                        pv = torch.tensor(pad_val, dtype=img.dtype, device=img.device).view(C, 1, 1)
                        out[:] = pv
                    out[:, padding[1]:padding[1] + H, padding[0]:padding[0] + W] = img
                    return out  # type: ignore[return-value]
            else:
                raise ValueError('Torch impad only supports 2D/3D tensors')
        else:
            mode_map = {
                'edge': 'replicate',
                'reflect': 'reflect',
                'symmetric': 'replicate',  # best-effort approximation
            }
            mode = mode_map[padding_mode]
            # F.pad expects (pad_l, pad_r, pad_t, pad_b) for 2D/3D (HWC not directly supported)
            if img.dim() == 2:
                return F.pad(img.unsqueeze(0).unsqueeze(0), (padding[0], padding[2], padding[1], padding[3]), mode=mode).squeeze(0).squeeze(0)  # type: ignore[return-value]
            elif img.dim() == 3:
                if img.shape[-1] <= 4:  # HWC -> CHW
                    x = img.permute(2, 0, 1).unsqueeze(0)
                    x = F.pad(x, (padding[0], padding[2], padding[1], padding[3]), mode=mode)
                    return x.squeeze(0).permute(1, 2, 0)  # type: ignore[return-value]
                else:  # CHW
                    x = img.unsqueeze(0)
                    x = F.pad(x, (padding[0], padding[2], padding[1], padding[3]), mode=mode)
                    return x.squeeze(0)  # type: ignore[return-value]
            else:
                raise ValueError('Torch impad only supports 2D/3D tensors')

    # Numpy/OpenCV branch
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


def impad_to_multiple(img: Union[np.ndarray, 'torch.Tensor'],
                      divisor: int,
                      pad_val: Union[float, List] = 0) -> Union[np.ndarray, 'torch.Tensor']:
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if img.dim() == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
        elif img.dim() == 3:
            if img.shape[-1] <= 4:
                h, w = int(img.shape[0]), int(img.shape[1])
            else:
                h, w = int(img.shape[1]), int(img.shape[2])
        else:
            raise ValueError('Torch impad_to_multiple supports 2D/3D tensors only')
        pad_h = int(np.ceil(h / divisor)) * divisor
        pad_w = int(np.ceil(w / divisor)) * divisor
        return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)  # type: ignore[return-value]

    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


def cutout(img: Union[np.ndarray, 'torch.Tensor'],
           shape: Union[int, Tuple[int, int]],
           pad_val: Union[int, float, tuple] = 0) -> Union[np.ndarray, 'torch.Tensor']:
    """Randomly cut out a rectangle from the original img.

    Args:
        img (ndarray): Image to be cutout.
        shape (int | tuple[int]): Expected cutout shape (h, w). If given as a
            int, the value will be used for both h and w.
        pad_val (int | float | tuple[int | float]): Values to be filled in the
            cut area. Defaults to 0.

    Returns:
        ndarray: The cutout image.
    """

    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        # Torch branch
        if img.dim() == 2:
            channels = 1
            img_h, img_w = int(img.shape[0]), int(img.shape[1])
        elif img.dim() == 3:
            if img.shape[-1] <= 4:
                channels = int(img.shape[2])
                img_h, img_w = int(img.shape[0]), int(img.shape[1])
            else:
                channels = int(img.shape[0])
                img_h, img_w = int(img.shape[1]), int(img.shape[2])
        else:
            raise ValueError('Torch cutout supports 2D/3D tensors only')

        if isinstance(shape, int):
            cut_h, cut_w = shape, shape
        else:
            assert isinstance(shape, tuple) and len(shape) == 2, \
                f'shape must be a int or a tuple with length 2, but got type {type(shape)} instead.'
            cut_h, cut_w = shape

        if isinstance(pad_val, (int, float)):
            pad_val_tuple = tuple([pad_val] * channels)
        elif isinstance(pad_val, tuple):
            assert len(pad_val) == channels, \
                'Expected the num of elements in tuple equals the channels' \
                f'of input image. Found {len(pad_val)} vs {channels}'
            pad_val_tuple = pad_val
        else:
            raise TypeError(f'Invalid type {type(pad_val)} for `pad_val`')

        y0 = torch.rand((), device=img.device) * img_h
        x0 = torch.rand((), device=img.device) * img_w

        y1 = int(max(0, float(y0.item()) - cut_h / 2.0))
        x1 = int(max(0, float(x0.item()) - cut_w / 2.0))
        y2 = min(img_h, y1 + cut_h)
        x2 = min(img_w, x1 + cut_w)

        if img.dim() == 2:
            patch_shape = (y2 - y1, x2 - x1)
            patch = torch.full(patch_shape, float(pad_val_tuple[0]), dtype=img.dtype, device=img.device)
            img_cutout = img.clone()
            img_cutout[y1:y2, x1:x2] = patch
            return img_cutout  # type: ignore[return-value]
        elif img.dim() == 3 and img.shape[-1] <= 4:
            patch_shape = (y2 - y1, x2 - x1, channels)
            pv = torch.tensor(pad_val_tuple, dtype=img.dtype, device=img.device).view(1, 1, channels)
            patch = pv.expand(patch_shape[0], patch_shape[1], channels).clone()
            img_cutout = img.clone()
            img_cutout[y1:y2, x1:x2, :] = patch
            return img_cutout  # type: ignore[return-value]
        else:
            patch_shape = (channels, y2 - y1, x2 - x1)
            pv = torch.tensor(pad_val_tuple, dtype=img.dtype, device=img.device).view(channels, 1, 1)
            patch = pv.expand(channels, patch_shape[1], patch_shape[2]).clone()
            img_cutout = img.clone()
            img_cutout[:, y1:y2, x1:x2] = patch
            return img_cutout  # type: ignore[return-value]

    # Numpy branch
    channels = 1 if img.ndim == 2 else img.shape[2]
    if isinstance(shape, int):
        cut_h, cut_w = shape, shape
    else:
        assert isinstance(shape, tuple) and len(shape) == 2, \
            f'shape must be a int or a tuple with length 2, but got type ' \
            f'{type(shape)} instead.'
        cut_h, cut_w = shape
    if isinstance(pad_val, (int, float)):
        pad_val = tuple([pad_val] * channels)
    elif isinstance(pad_val, tuple):
        assert len(pad_val) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(pad_val), channels)
    else:
        raise TypeError(f'Invalid type {type(pad_val)} for `pad_val`')

    img_h, img_w = img.shape[:2]
    y0 = np.random.uniform(img_h)
    x0 = np.random.uniform(img_w)

    y1 = int(max(0, y0 - cut_h / 2.))
    x1 = int(max(0, x0 - cut_w / 2.))
    y2 = min(img_h, y1 + cut_h)
    x2 = min(img_w, x1 + cut_w)

    if img.ndim == 2:
        patch_shape = (y2 - y1, x2 - x1)
    else:
        patch_shape = (y2 - y1, x2 - x1, channels)  # type: ignore

    img_cutout = img.copy()
    patch = np.array(
        pad_val, dtype=img.dtype) * np.ones(
            patch_shape, dtype=img.dtype)
    img_cutout[y1:y2, x1:x2, ...] = patch

    return img_cutout


def _get_shear_matrix(magnitude: Union[int, float],
                      direction: str = 'horizontal') -> np.ndarray:
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        ndarray: The shear matrix with dtype float32.
    """
    if direction == 'horizontal':
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == 'vertical':
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix


def imshear(img: Union[np.ndarray, 'torch.Tensor'],
            magnitude: Union[int, float],
            direction: str = 'horizontal',
            border_value: Union[int, Tuple[int, int]] = 0,
            interpolation: str = 'bilinear') -> Union[np.ndarray, 'torch.Tensor']:
    """Shear an image.

    Args:
        img (ndarray): Image to be sheared with format (h, w)
            or (h, w, c).
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The sheared image.
    """
    assert direction in ['horizontal', 'vertical'], f'Invalid direction: {direction}'
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        # Prepare BCHW
        if kornia_warp_affine is None:
            raise RuntimeError('kornia is required for torch-based imshear. Please install kornia.')
        if img.dim() == 2:
            H, W = int(img.shape[0]), int(img.shape[1])
            x = img.unsqueeze(0).unsqueeze(0)
            layout = 'HW'
        elif img.dim() == 3:
            if img.shape[-1] <= 4:  # HWC
                H, W, C = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                x = img.permute(2, 0, 1).unsqueeze(0)
                layout = 'HWC'
            else:  # CHW
                C, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                x = img.unsqueeze(0)
                layout = 'CHW'
        else:
            raise ValueError('Torch imshear supports 2D/3D tensors only')

        shear_matrix = torch.tensor(
            [[1, magnitude, 0], [0, 1, 0]] if direction == 'horizontal' else [[1, 0, 0], [magnitude, 1, 0]],
            dtype=torch.float32, device=x.device
        ).unsqueeze(0)

        interp = 'bilinear' if interpolation not in ('nearest', 'bilinear') else interpolation
        padding_mode = 'zeros'
        # ensure float for warp
        orig_dtype = x.dtype
        need_float = not torch.is_floating_point(x)
        x_in = x.to(torch.float32) if need_float else x
        try:
            y = kornia_warp_affine(x_in, shear_matrix, dsize=(H, W), mode=interp, padding_mode=padding_mode, align_corners=False, fill_value=border_value)
        except TypeError:
            y = kornia_warp_affine(x_in, shear_matrix, dsize=(H, W), mode=interp, padding_mode=padding_mode, align_corners=False)
        if need_float:
            y = y.round().to(orig_dtype)

        if layout == 'HW':
            return y.squeeze(0).squeeze(0).to(img.dtype)  # type: ignore[return-value]
        elif layout == 'HWC':
            return y.squeeze(0).permute(1, 2, 0).to(img.dtype)  # type: ignore[return-value]
        else:
            return y.squeeze(0).to(img.dtype)  # type: ignore[return-value]

    # Numpy/OpenCV branch
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)  # type: ignore
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`')
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(
        img,
        shear_matrix,
        (width, height),
        borderValue=border_value[:3],  # type: ignore
        flags=cv2_interp_codes[interpolation])
    return sheared


def _get_translate_matrix(offset: Union[int, float],
                          direction: str = 'horizontal') -> np.ndarray:
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == 'horizontal':
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == 'vertical':
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(img: Union[np.ndarray, 'torch.Tensor'],
                offset: Union[int, float],
                direction: str = 'horizontal',
                border_value: Union[int, tuple] = 0,
                interpolation: str = 'bilinear') -> Union[np.ndarray, 'torch.Tensor']:
    """Translate an image.

    Args:
        img (ndarray): Image to be translated with format
            (h, w) or (h, w, c).
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The translated image.
    """
    assert direction in ['horizontal', 'vertical'], f'Invalid direction: {direction}'
    if 'torch' in globals() and torch is not None and isinstance(img, torch.Tensor):
        if kornia_warp_affine is None:
            raise RuntimeError('kornia is required for torch-based imtranslate. Please install kornia.')
        # Prepare BCHW
        if img.dim() == 2:
            H, W = int(img.shape[0]), int(img.shape[1])
            x = img.unsqueeze(0).unsqueeze(0)
            layout = 'HW'
        elif img.dim() == 3:
            if img.shape[-1] <= 4:  # HWC
                H, W, C = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                x = img.permute(2, 0, 1).unsqueeze(0)
                layout = 'HWC'
            else:  # CHW
                C, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                x = img.unsqueeze(0)
                layout = 'CHW'
        else:
            raise ValueError('Torch imtranslate supports 2D/3D tensors only')

        if isinstance(border_value, int):
            border_value_tuple = (border_value,)
        elif isinstance(border_value, tuple):
            border_value_tuple = border_value
        else:
            raise ValueError(f'Invalid type {type(border_value)} for `border_value`.')

        translate_matrix = torch.tensor(
            [[1, 0, offset], [0, 1, 0]] if direction == 'horizontal' else [[1, 0, 0], [0, 1, offset]],
            dtype=torch.float32, device=x.device
        ).unsqueeze(0)
        interp = 'bilinear' if interpolation not in ('nearest', 'bilinear') else interpolation
        # ensure float for warp
        orig_dtype = x.dtype
        need_float = not torch.is_floating_point(x)
        x_in = x.to(torch.float32) if need_float else x
        try:
            y = kornia_warp_affine(x_in, translate_matrix, dsize=(H, W), mode=interp, padding_mode='zeros', align_corners=False, fill_value=border_value_tuple[0])
        except TypeError:
            y = kornia_warp_affine(x_in, translate_matrix, dsize=(H, W), mode=interp, padding_mode='zeros', align_corners=False)
        if need_float:
            y = y.round().to(orig_dtype)

        if layout == 'HW':
            return y.squeeze(0).squeeze(0).to(img.dtype)  # type: ignore[return-value]
        elif layout == 'HWC':
            return y.squeeze(0).permute(1, 2, 0).to(img.dtype)  # type: ignore[return-value]
        else:
            return y.squeeze(0).to(img.dtype)  # type: ignore[return-value]

    # Numpy/OpenCV branch
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`.')
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(
        img,
        translate_matrix,
        (width, height),
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation])
    return translated
