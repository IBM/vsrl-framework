#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from typing import Union

import cv2
import numpy as np
from PIL import Image


def make_video(filename: str, frames, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out: cv2.VideoWriter = cv2.VideoWriter(filename, fourcc, 30, (width, height))
    try:
        for frame in frames:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB))
    finally:
        # in case of a KeyboardInterrupt or other error, at lease save the frames that were generated.
        out.release()


def show_image(image: object) -> None:
    if isinstance(image, np.ndarray):
        Image.fromarray(image).show()
    else:
        image.show()


def paste_coordinates(img: object, x: Union[int, float], y: Union[int, float]):
    """
    Pillow uses the top left corner for pasting images, but our models often refer to the midpoints of objects.
    This function takes in the x,y coordinates where we want the image's midpoint to be, and returns the x,y
    coordinates that should be passed into pilloe's paste function in order to center the image at x,y.
    :param img: The image, needed so that we can get its size.
    :param x: The desired midpoint of the pasted image (x coordinate)
    :param y: The desired midpoint of the pasted image (y coordinate)
    :return: x,y
    """
    return (int(x - img.size[0] / 2), int(y - img.size[1] / 2))
