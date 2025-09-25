import lmdb
import os
from typing import List, Optional


def create_rawimage_dataset(output_path: str,
                            image_file_list: List[str],
                            image_tmpl: Optional[str] = None,
                            flag: str = 'color',
                            check_valid: bool = True):
    """Create an LMDB dataset that stores raw image bytes.

    Args:
        output_path: LMDB directory to create.
        image_file_list: List of file paths to images.
        image_tmpl: Optional format string for keys; if None, uses basename.
        flag: Unused in this simple creator; for compatibility.
        check_valid: If True, skip unreadable files.
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1024 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        for idx, fp in enumerate(sorted(image_file_list)):
            try:
                with open(fp, 'rb') as f:
                    img_bytes = f.read()
            except Exception:
                if check_valid:
                    continue
                else:
                    raise
            base = os.path.basename(fp)
            if image_tmpl is None:
                key = base
            else:
                try:
                    # Support templates like 'img_{:05d}' or '{:s}'
                    key = image_tmpl.format(idx)
                except Exception:
                    key = base
            txn.put(str(key).encode(), img_bytes)
    env.sync()
    env.close()
