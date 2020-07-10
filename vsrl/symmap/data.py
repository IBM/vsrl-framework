#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from math import ceil
from pathlib import Path
from random import choice, randint, random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import toml
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomCrop, ToTensor
from torchvision.transforms.functional import resized_crop


def parse_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a configuration file for object detection into a Python dictionary.

    The format of the config file is
    ```toml
    backgrounds = [] # list of paths to background images
    [objects]
    obj1 = [] # list of paths to images of object named "obj1"
    obj2 = [] # list of paths to images of object named "obj2"
    # ... any number of objects may be given
    ```
    If relative paths are used, they're resolved as being relative to the directory
    that the config file is in.
    """
    config_path = Path(config_path).resolve()
    config = toml.load(config_path)
    resolve_paths(config, config_path.parent)
    return config


def resolve_paths(config: Dict[str, Any], config_dir: Path) -> None:
    """
    Convert relative paths to absolute ones (in-place).
    :param config_dir: all paths are relative to this directory
    """
    for path_list in [config["backgrounds"], *config["objects"].values()]:
        for i, path in enumerate(path_list):
            path = Path(path)
            if not path.is_absolute():
                path = (config_dir / path).resolve()
                path_list[i] = str(path)


def make_2d_gaussian(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    var2x: float,
    var2y: float,
    output_scale: int = 1,
) -> np.ndarray:
    """

    (0, 0) is the top-left corner.
    :param var2x: 2 * variance_x
    :param var2y: 2 * variance_y
    :param output_scale: all of the other parameters will be divided by this (width,
      height, center_x, and center_y are floored after this division)
    :returns: (height x width)
    """
    if output_scale > 1:
        width = int(ceil(width / output_scale))
        height = int(ceil(height / output_scale))
        center_x //= output_scale
        center_y //= output_scale
        var2x /= output_scale
        var2y /= output_scale
    else:
        # ensure these are ints so that there's a cell with a 1 in the output
        center_x //= 1
        center_y //= 1
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return np.exp(-((x - center_x) ** 2 / var2x + (y - center_y) ** 2 / var2y))


def gen_img(
    bg_imgs,
    obj_imgs,
    resize_shape=None,
    expected_n_objs: int = 10,
    lr_flip_prob: float = 0.5,
    crop_prob: float = 0.75,
    output_scale: int = 4,
) -> Tuple[Image.Image, Dict[str, np.ndarray]]:
    """
    :param output_scale: how many times smaller the output of the model is than the original
      image.
    :returns: (img, labels) where
    * img is an image from `bg_imgs` with objects from `obj_imgs` pasted in
    * labels has
      * probs (k x h x w) for k-way classification (k objects in `obj_imgs`)
      * offsets (2 x h x w) where the 2 are (x_offset, y_offset)
    """
    bg_img = choice(bg_imgs).copy()
    bg_width, bg_height = bg_img.size

    if resize_shape:
        crop = RandomCrop(resize_shape)
    else:
        crop = None
        resize_shape = (bg_height, bg_width)

    if lr_flip_prob < random():
        bg_img = bg_img.transpose(Image.FLIP_LEFT_RIGHT)

    # I crop the background instead of the scene so that no objects have their centers outside of
    # the image. CenterNet or CenterTrack does something about such objects, but I'm not sure what
    if random() < crop_prob:
        min_width = int(0.6 * bg_width)
        min_height = int(0.6 * bg_height)
        crop_y = randint(0, bg_height - min_height)
        crop_x = randint(0, bg_width - min_width)
        crop_height = randint(min_height, bg_height - crop_y)
        crop_width = randint(min_width, bg_width - crop_x)
        bg_img = resized_crop(
            bg_img, crop_y, crop_x, crop_height, crop_width, resize_shape
        )
    elif crop:
        bg_img = crop(bg_img)

    bg_width, bg_height = bg_img.size

    n_objs = sum(map(len, obj_imgs.values()))
    # might have to do multiple passes to get enough objects on average
    n_passes = int(ceil(1.5 * expected_n_objs / n_objs))
    obj_prob = expected_n_objs / (n_objs * n_passes)

    label = []
    offset = np.zeros(
        (2, ceil(bg_height / output_scale), ceil(bg_width / output_scale)),
        dtype=np.float32,
    )
    for img_list in obj_imgs.values():
        obj_label = []
        for _ in range(n_passes):
            for img in img_list:
                if random() > obj_prob:
                    continue

                if lr_flip_prob < random():
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                img = img.rotate(randint(0, 359))

                # pick insertion point for the top-left corner

                insert_x = randint(-img.width // 2, bg_width - img.width // 2 - 1)
                insert_y = randint(-img.height // 2, bg_height - img.height // 2 - 1)
                # if isinstance(template, CombinedTemplate):
                # too easy to have one template be too far offscreen
                # insert_x = randint(0, bg_width - width)
                # insert_y = randint(0, bg_height - height)
                center_x = insert_x + img.width / 2
                center_y = insert_y + img.height / 2

                ## handle template being partially off-screen
                if insert_x < 0:
                    img = img.crop((abs(insert_x), 0, img.width, img.height))
                    insert_x = 0
                elif insert_x + img.width > bg_width:
                    img = img.crop((0, 0, bg_width - insert_x, img.height))
                    # img = img.crop(0, 0, width - (insert_x + width - bg_width), height)
                    insert_x = bg_width - img.width

                if insert_y < 0:
                    img = img.crop((0, abs(insert_y), img.width, img.height))
                    insert_y = 0
                elif insert_y + img.height > bg_height:
                    img = img.crop((0, 0, img.width, bg_height - insert_y))
                    insert_y = bg_height - img.height

                x_scaled_floored = int(center_x / output_scale)
                y_scaled_floored = int(center_y / output_scale)
                offset_x = center_x / output_scale - x_scaled_floored
                offset_y = center_y / output_scale - y_scaled_floored
                if (offset[:, y_scaled_floored, x_scaled_floored] > 0).any():
                    # if there are multiple objects with the same center, just average
                    # the offset prediction. Technically, this is only an average if
                    # there are 2 objects, but 3 or more with the same center is probably
                    # very rare
                    offset[:, y_scaled_floored, x_scaled_floored] += offset_x, offset_y
                    offset[:, y_scaled_floored, x_scaled_floored] /= 2
                offset[:, y_scaled_floored, x_scaled_floored] = offset_x, offset_y

                img_label = make_2d_gaussian(
                    bg_width,
                    bg_height,
                    center_x,
                    center_y,
                    img.width,
                    img.height,
                    output_scale,
                )

                obj_label.append(img_label)
                bg_img.paste(img, (insert_x, insert_y), mask=img)
        if obj_label:
            obj_label = np.max(np.stack(obj_label), 0)
        else:
            obj_label = np.zeros(
                (
                    int(ceil(bg_height / output_scale)),
                    int(ceil(bg_width / output_scale)),
                ),
                dtype=np.float32,
            )
        label.append(obj_label)
    return bg_img, {"probs": np.stack(label, 0), "offset": offset}


class ImgDataset(Dataset):
    def __init__(
        self,
        bg_imgs,
        obj_imgs,
        epoch_size: int,
        label_scale: int = 4,
        transform=None,
        grayscale: bool = False,
        img_scale: int = 1,
        img_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        :param label_scale: how much to downscale the labels compared to the input images
          An input of 4 means the labels are scaled down 4x (i.e. 1/4 the img size).
        :param img_scale: how much to scale down the backgrounds / objects before
          generating images.
        :param img_shape: (height, width) the shape that the generated images should be
          resized to. If `None`, the size of the smallest background image (after
          downscaling) will be used (height * width is used to determine the "smallest").
        """
        if img_scale > 1:
            self.bg_imgs = [img.reduce(img_scale) for img in bg_imgs]
            self.obj_imgs = {
                k: [img.reduce(img_scale) for img in v] for k, v in obj_imgs.items()
            }
        else:
            self.bg_imgs = bg_imgs
            self.obj_imgs = obj_imgs

        if img_shape is None:
            img_shape = min(
                [img.size for img in self.bg_imgs], key=lambda s: s[0] * s[1]
            )
            img_shape = tuple(reversed(img_shape))  # PIL Images have (width, height)

        self.epoch_size = epoch_size
        self.label_scale = label_scale
        self.graycsale = grayscale
        self.img_scale = img_scale
        self.img_shape = img_shape
        if transform:
            self.transform = transform
        else:
            if self.graycsale:
                # L for luminosity, i.e. grayscale
                self.transform = Compose([lambda img: img.convert("L"), ToTensor()])
            else:
                self.transform = Compose([ToTensor(), drop_alpha])

    def __getitem__(self, index: int):
        img, label = gen_img(
            self.bg_imgs, self.obj_imgs, self.img_shape, output_scale=self.label_scale,
        )
        img = self.transform(img)
        return img, label

    def gen_img(self):
        """Returns raw_img, img, label; useful for debugging."""
        raw_img, label = gen_img(
            self.bg_imgs, self.obj_imgs, self.img_shape, output_scale=self.label_scale,
        )
        img = self.transform(raw_img)
        return raw_img, img, label

    def __len__(self):
        return self.epoch_size


def drop_alpha(img: torch.Tensor) -> torch.Tensor:
    """img must be CHW"""
    return img[:3]


class DummyDataset(Dataset):
    """A dataset with only one input-output example for debugging."""

    def __init__(self, x, y, epoch_size):
        self.x = x
        self.y = y
        self.epoch_size = epoch_size

    def __getitem__(self, index: int):
        return self.x, self.y

    def __len__(self):
        return self.epoch_size
