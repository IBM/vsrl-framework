#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import argparse
import logging
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def to_grayscale(img: np.array, msg: str = None):
    """
    Convert an image to grayscale.

    :param img: The image to convert.
    :param msg: a message to print out to the WARN log, or else None if the conversion should happen without warning.

    :return: The image converted to grayscale, or the original image.
    """
    if len(img.shape) > 2:
        if msg is not None:
            logging.warning(msg)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def replace_simultaneous(s: str, r0: str, r1: str) -> str:
    """
    Replace all instances of `r0` in `s` with `r1` and vice versa.

    This method does the replacements simultaneously so that, e.g.,
    if you call `replace_simultaneous("apple banana", "apple", "banana")`
    the result will be `"banana apple"` instead of either `"apple apple"`
    or `"banana banana"` for a sequential replacement (depending on the
    order of the replacements).
    It is assumed that `r0` is not a substring of `r1` and vice versa.

    :param s: string to do replacements in
    :param r0: first string to replace
    :param r1: second string to replace
    """
    replacement = {r0: r1, r1: r0}
    r0 = re.escape(r0)
    r1 = re.escape(r1)
    return re.sub(f"{r0}|{r1}", lambda x: replacement[x.group()], s)


def remove_objs(
    img: np.ndarray,
    obj_masks: Union[np.ndarray, Sequence[np.ndarray]],
    obj_locs: Union[Sequence[int], Sequence[Sequence[int]]],
) -> np.ndarray:
    """
    :param obj_masks: shape (w x h)
    :param obj_locs: (x, y, w, h)
    """
    if not isinstance(obj_locs[0], Sequence):
        obj_locs = [obj_locs]
        obj_masks = [obj_masks]
    for loc, mask in zip(obj_locs, obj_masks):
        x, y, w, h = loc
        img_mask = np.zeros((img.shape[:-1]), dtype=np.uint8)
        img_mask[y : y + h, x : x + w] = mask
        img = cv2.inpaint(img, img_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    return img


def make_2d_gaussian(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    var2x: float,
    var2y: float,
) -> np.ndarray:
    """
    :param var2x: 2 * variance_x
    :param var2y: 2 * variance_y
    :returns: (height x width)
    """
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return np.exp(-((x - center_x) ** 2 / var2x + (y - center_y) ** 2 / var2y))


def draw_rects(
    img: np.ndarray,
    x: Union[int, Iterable[int]],
    y: Union[int, Iterable[int]],
    width: Union[int, Iterable[int]],
    height: Union[int, Iterable[int]],
    color: int = 0,
    thickness: int = 1,
    copy: bool = False,
) -> np.ndarray:
    """
    :param img: the original image
    :param x: the x position(s) of the box(s).
    :param y: if iterable, must have the same length as x
    :param width: if not iterable, same value used for all (x, y)
    :param height: if not iterable, same value used for all (x, y)
    """
    xs = x if isinstance(x, Iterable) else [x]
    ys = y if isinstance(y, Iterable) else [y]
    widths = width if isinstance(width, Iterable) else [width] * len(xs)
    heights = height if isinstance(height, Iterable) else [height] * len(xs)
    if copy:
        img = img.copy()
    for x, y, width, height in zip(xs, ys, widths, heights):
        img[y : y + height + 1, x : x + thickness] = color
        img[y : y + height + 1, x + width : x + width + thickness] = color
        img[y : y + thickness, x : x + width + 1] = color
        img[y + height : y + height + thickness, x : x + width + 1] = color
    return img


def drop_zeros(
    img: np.ndarray, mask: Optional[np.ndarray] = None, return_info: bool = False
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, Dict[str, int]],
    Tuple[np.ndarray, np.ndarray, Dict[str, int]],
]:
    """
    Drop initial and final rows / columns that are all 0.
    The input arrays are assumed to have even height and width. The output arrays will
    too, including keeping a final row or column that is all 0s if necessary.
    :param img: (height x width x channels)
    :param mask: same height and width as img
    :param return_info: whether to return how many rows / cols were dropped
    :returns: img if mask is None else (img, mask)
    """
    all0_rows = (img == 0).all((1, 2)).nonzero()[0]
    all0_cols = (img == 0).all((0, 2)).nonzero()[0]

    height, width = img.shape[:-1]

    # find number of consecutive indices at start / end
    n_drop_top = (np.arange(len(all0_rows)) == all0_rows).sum()
    n_drop_left = (np.arange(len(all0_cols)) == all0_cols).sum()
    reverse_idx = np.arange(height - 1, height - len(all0_rows) - 1, -1)[::-1]
    n_drop_bot = (reverse_idx == all0_rows).sum()
    reverse_idx = np.arange(width - 1, width - len(all0_cols) - 1, -1)[::-1]
    n_drop_right = (reverse_idx == all0_cols).sum()

    # keep one extra row / col if necessary for even height / width
    n_drop_bot -= (height - n_drop_top - n_drop_bot) % 2
    n_drop_right -= (width - n_drop_left - n_drop_right) % 2
    info = {
        "top": n_drop_top,
        "right": n_drop_right,
        "bot": n_drop_bot,
        "left": n_drop_left,
    }
    img = img[n_drop_top : height - n_drop_bot, n_drop_left : width - n_drop_right]
    if mask is not None:
        mask = mask[
            n_drop_top : height - n_drop_bot, n_drop_left : width - n_drop_right
        ]
        return (img, mask, info) if return_info else (img, mask)
    return (img, info) if return_info else img


def str2bool(v):
    """Convert a string into a Boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
