"""
Local compatibility package for mmcv.

Bridges to vendored mmcv modules located under mmcv/mmcv and exposes
missing legacy namespaces (runner, parallel, lmdb) implemented via
mmengine/torch.
"""
import sys as _sys
from mmengine.utils import is_str  # re-export helper

# Provide compatibility submodules (implemented locally)
from . import runner as runner  # noqa: F401
from . import parallel as parallel  # noqa: F401
from . import lmdb as lmdb  # noqa: F401

# Re-export and expose our local minimal subpackages
from .image import imfrombytes, imread, imwrite  # noqa: F401
from . import image as image  # noqa: F401
from . import video as video  # noqa: F401
from . import visualization as visualization  # noqa: F401
from . import cnn as cnn  # noqa: F401

_sys.modules.setdefault(__name__ + '.image', image)
_sys.modules.setdefault(__name__ + '.video', video)
_sys.modules.setdefault(__name__ + '.visualization', visualization)
_sys.modules.setdefault(__name__ + '.cnn', cnn)
