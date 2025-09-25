import io as _io
from pathlib import Path as _Path
from typing import Optional as _Optional, Union as _Union

import cv2 as _cv2
import numpy as _np


IMREAD_COLOR = _cv2.IMREAD_COLOR


def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: _Optional[str] = None) -> _np.ndarray:
    if backend is not None and backend.lower() not in (None, 'cv2'):
        raise ValueError('Only cv2 backend is supported in this shim')
    img_np = _np.frombuffer(content, _np.uint8)
    flag_map = dict(
        color=IMREAD_COLOR,
        grayscale=_cv2.IMREAD_GRAYSCALE,
        unchanged=_cv2.IMREAD_UNCHANGED,
    )
    cv_flag = flag_map[flag] if isinstance(flag, str) else flag
    img = _cv2.imdecode(img_np, cv_flag)
    if cv_flag == IMREAD_COLOR and channel_order == 'rgb':
        _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB, img)
    return img


def imread(img_or_path: _Union[str, _np.ndarray, _Path],
           flag: str = 'color',
           channel_order: str = 'bgr') -> _np.ndarray:
    if isinstance(img_or_path, _Path):
        img_or_path = str(img_or_path)
    if isinstance(img_or_path, _np.ndarray):
        return img_or_path
    if isinstance(img_or_path, str):
        with open(img_or_path, 'rb') as f:
            img_bytes = f.read()
        return imfrombytes(img_bytes, flag, channel_order, backend='cv2')
    raise TypeError('"img" must be a numpy array or a str or a pathlib.Path')


def imwrite(img: _np.ndarray, file_path: str) -> bool:
    return _cv2.imwrite(file_path, img)

