#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import holoviews as hv
import numpy as np
from IPython.display import display

import vsrl.symmap.symbolic_mapper
import vsrl.verifier.expr as vexpr
from vsrl.spaces.space import Space
from vsrl.symmap.utils import draw_rects, drop_zeros, to_grayscale


class TemplateMatching(vsrl.symmap.symbolic_mapper.SymbolicMapper):
    """
    TemplateMatching uses cv2's built-in template matching function.
    Images are always converted into grayscale, including both the input image and the template images. This method
    """

    def __init__(
        self,
        templates: Iterable[np.ndarray],
        space: Space,
        threshold: float = -1,
        corr: str = "TM_CCORR_NORMED",
    ):
        """
        :param templates: grayscale templates. Will convert to grayscale if they are not already in grawscale.
        :param threshold: threshold at or above which a match is detected.
        If threshold < 0, only the match with highest confidence will be returned.
        :param corr: comparison method. See the OpenCV cv2.matchTemplate() documentation for options and details.
        """
        for i, t in enumerate(templates):
            templates[i] = to_grayscale(t)
        self.templates = templates
        self.threshold = threshold
        self.corr = corr
        self._space = space

    @property
    def space(self) -> Space:
        return self._space

    def _raw_map(self, raw_img: np.ndarray):
        return self._match_templates(
            to_grayscale(raw_img), self.templates, self.threshold, self.corr
        )

    def __call__(self, raw_img: np.ndarray) -> np.ndarray:
        values = self._raw_map(raw_img)
        s = [item for sublist in values for coords in sublist for item in coords]
        assert np.array(s) in self.space
        return np.array(s)

    def draw_bounding_boxes(self, raw_img: np.ndarray, bb_color: int = 0):
        positions = self._raw_map(raw_img)
        raw_img = to_grayscale(raw_img)
        for i, template in enumerate(self.templates):
            h, w = template.shape
            for y, x in positions[i]:
                raw_img = draw_rects(raw_img, x, y, w, h, bb_color, 1, False)
        return raw_img

    @staticmethod
    def _match_templates(
        raw_img: np.ndarray,
        templates: Iterable[np.ndarray],
        threshold: float,
        corr: str = "TM_CCORR_NORMED",
    ) -> List[Tuple[float, float]]:
        """
        :param raw_img: RGB or grayscale image; converted to grayscale if RGB
        :param templates: grayscale templates
        :param threshold: threshold at or above which a match is detected.
        If threshold < 0, only the match with highest confidence will be returned.
        :param corr: comparison method. See the OpenCV cv2.matchTemplate() documentation for options and details.
        """
        img = (
            raw_img if raw_img.ndim == 2 else cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        )
        all_match_locs = []
        for template in templates:
            template_match = cv2.matchTemplate(img, template, getattr(cv2, corr))
            if corr == "TM_SQDIFF_NORMED":
                if threshold < 0:
                    match_locs = [
                        np.unravel_index(
                            np.argmin(template_match), template_match.shape
                        )
                    ]
                else:
                    match_locs = np.where(template_match <= 1 - threshold)
            else:
                if threshold < 0:
                    match_locs = [
                        np.unravel_index(
                            np.argmax(template_match), template_match.shape
                        )
                    ]
                else:
                    match_locs = np.where(template_match >= threshold)

            if threshold < 0:
                all_match_locs.append(match_locs)
            else:
                all_match_locs.append(list(zip(*match_locs)))

        return all_match_locs


def make_template_from_img(
    img: np.ndarray, x: int, y: int, width: int, height: int,
) -> np.ndarray:
    template = img[y : y + height, x : x + width]
    return template
