from typing import List, Optional, Union

import cv2
import numpy as np

from mmcv.image import imread, imwrite

ColorType = Union[str, tuple, int, np.ndarray]


def imshow_det_bboxes(img: Union[str, np.ndarray],
                      bboxes: np.ndarray,
                      labels: np.ndarray,
                      class_names: List[str] = None,
                      score_thr: float = 0,
                      bbox_color: ColorType = 'green',
                      text_color: ColorType = 'green',
                      thickness: int = 1,
                      font_scale: float = 0.5,
                      show: bool = True,
                      win_name: str = '',
                      wait_time: int = 0,
                      out_file: Optional[str] = None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] in (4, 5)
    img = imread(img)
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), thickness=thickness)
        label_text = class_names[label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 255, 0))

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img

