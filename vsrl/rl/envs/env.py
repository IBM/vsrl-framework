#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from abc import ABC, abstractmethod
from math import isnan, pi
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import gym
import numpy as np
from PIL import Image

from vsrl.rl.envs.render_helpers import paste_coordinates
from vsrl.spaces.space import Space


class Observation(NamedTuple):
    img: np.ndarray
    vector: np.ndarray  # any non-image (1D) features (velocity, angle)


class MaskedImg(NamedTuple):
    img: Image.Image
    mask: Image.Image


_T = TypeVar("_T")
_TIdx = Union[int, float]
_TMaybeSeq = Union[_T, Sequence[_T]]
_TObjsInit = Dict[
    str,
    Union[
        Tuple[Image.Image, _TMaybeSeq[_TIdx], _TMaybeSeq[_TIdx]],
        Tuple[
            Image.Image,
            _TMaybeSeq[_TIdx],
            _TMaybeSeq[_TIdx],
            _TMaybeSeq[Optional[_TIdx]],
        ],
    ],
]
_TObjs = Dict[
    str, Tuple[MaskedImg, Sequence[_TIdx], Sequence[_TIdx], Sequence[Optional[_TIdx]]]
]


class Env(ABC, gym.Env):
    """
    VSRL environments are bit different from environments in other RL frameworks.

    A VSRL environment has two views on its state: a raw state and an oracle state.
    The raw state is the classic state; for example, a pixel map.
    The oracle state is a symbolic state; for example, the positions and velocities of various objects.
    """

    def __init__(
        self,
        img_scale: int,
        grayscale: bool,
        oracle_obs: bool,
        scene: Image.Image,
        objs: _TObjsInit,
        horizon: int = 1000,
        vector_obs_bounds: Optional[Tuple[List[int], List[int]]] = None,
    ):
        """
        :param img_scale: how much to scale the images (background + objects) down by.
          If 1, no scaling is used. If, e.g., 2, the images are half their usual size.
        :param graycsale: if True, all images (and thus observations) are converted to
          grayscale.
        :param oracle_obs: whether to return oracle observations (if True) or images
        :param objs: {object_name: (img, x_idx, y_idx)} where the indices are into the
          state vector, i.e. state[x_idx] gives the x location where the image should be
          pasted when rendering the object. If either index is a float instead of an int,
          it is used directly as the coordinate instead of as an index into the state vector.
        :param vector_obs_bounds: lower and upper bounds on each dimension of the vector
          part of the observations. If given, observations are named tuples with obs.img
          and obs.vector; a non-`None` `vector_obs` must always be given when `_get_obs`
          is called. If not given, observations are numpy arrays (just images) and
          `vector_obs` must always be `Non` when calling `_get_obs`.

        Because the images are converted at init (to make rendering faster) grayscale /
        img_scale can't be changed later.
        """
        self._img_scale = img_scale
        self._grayscale = grayscale
        self.oracle_obs = oracle_obs
        self.horizon = horizon

        self._scene, self._objs = self._preprocess_imgs(scene, objs)
        self._width, self._height = self._scene.size
        c = 1 if grayscale else 3
        self._prev_frame = np.zeros((self._height, self._width, c), dtype=np.uint8)
        self._oracle_space = self._make_oracle_space()
        self._action_space = self._make_action_space()
        # use float32 for the spaces to avoid a warning from gym
        self.action_space = gym.spaces.Box(
            self._action_space.lower_bounds.astype(np.float32),
            self._action_space.upper_bounds.astype(np.float32),
        )
        # this state doesn't have to be entirely correct; reset() is called before step
        self._state = self.oracle_space.sample().astype(np.float32)
        self._done = True
        self._step = 0

        if oracle_obs:
            self.observation_space = gym.spaces.Box(
                self.oracle_space.lower_bounds.astype(np.float32),
                self.oracle_space.upper_bounds.astype(np.float32),
            )
        else:
            img_obs_space = gym.spaces.Box(
                np.float32(0), np.float32(255), shape=(self._height, self._width, 2 * c)
            )
            if vector_obs_bounds is not None:
                vector_obs_space = gym.spaces.Box(
                    *[np.array(bnd, dtype=np.float32) for bnd in vector_obs_bounds]
                )
                self.observation_space = gym.spaces.Dict(
                    {"img": img_obs_space, "vector": vector_obs_space}
                )
            else:
                self.observation_space = img_obs_space

    def _preprocess_imgs(
        self, scene: Image.Image, objs: _TObjsInit
    ) -> Tuple[Image.Image, _TObjs]:
        processed_objs = {}
        for name, (img, x_idxs, y_idxs, *maybe_theta_idxs) in objs.items():
            img.load()
            if self._img_scale > 1:
                img = img.reduce(self._img_scale)
            mask = img.split()[-1]  # alpha channel
            # L = luminosity
            img = img.convert("L") if self._grayscale else img.convert("RGB")

            if not isinstance(x_idxs, Sequence):
                x_idxs = (x_idxs,)
            if not isinstance(y_idxs, Sequence):
                y_idxs = (y_idxs,)
            if maybe_theta_idxs:
                maybe_theta_idxs = maybe_theta_idxs[0]
                theta_idxs = (
                    maybe_theta_idxs
                    if isinstance(maybe_theta_idxs, Sequence)
                    else (maybe_theta_idxs,)
                )
            else:  # no thetas provided
                theta_idxs = (None,) * len(x_idxs)

            processed_objs[name] = (MaskedImg(img, mask), x_idxs, y_idxs, theta_idxs)

        scene.load()
        if self._img_scale > 1:
            scene = scene.reduce(self._img_scale)
        scene = scene.convert("L") if self._grayscale else scene.convert("RGB")
        return scene, processed_objs

    def _get_obs(
        self, vector_obs: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Observation]:
        """
        This alters self._prev_frame, so make sure it's only called once per step.
        """
        if self.oracle_obs:
            return self.current_oracle_state()

        current_frame = self.current_raw_state()
        img_obs = np.concatenate((self._prev_frame, current_frame), -1)
        self._prev_frame = current_frame
        if vector_obs is not None:
            return Observation(img_obs, vector_obs)
        return img_obs

    def current_raw_state(self) -> np.ndarray:
        """Returns uint8 array with dimensions HWC."""
        img = np.array(self.render())
        return img.reshape(self._height, self._width, -1)

    def render(self, mode: str = "") -> Union[Image.Image, np.ndarray]:
        """
        :param mode: if "array" a numpy array is returned; otherwise, a PIL Image.
        """
        scene = self._scene.copy()
        for (masked_img, x_idxs, y_idxs, theta_idxs) in self._objs.values():
            for x_idx, y_idx, theta_idx in zip(x_idxs, y_idxs, theta_idxs):
                x = self._state[x_idx] if isinstance(x_idx, int) else int(x_idx)
                y = self._state[y_idx] if isinstance(y_idx, int) else int(y_idx)
                if theta_idx is not None:
                    theta = (
                        self._state[theta_idx] * 180 / pi
                        if isinstance(theta_idx, int)
                        else theta_idx
                    )
                    masked_img = MaskedImg(
                        masked_img.img.rotate(theta), masked_img.mask.rotate(theta)
                    )

                if isnan(x) or isnan(y):
                    continue
                scene.paste(
                    masked_img.img,
                    paste_coordinates(masked_img.img, x, y),
                    mask=masked_img.mask,
                )
        if mode == "array":
            return np.array(scene)
        return scene

    @abstractmethod
    def _make_oracle_space(self) -> Space:
        raise NotImplementedError

    @abstractmethod
    def _make_action_space(self) -> Space:
        raise NotImplementedError

    @property
    def oracle_space(self) -> Space:
        """
        The observation space is the symbolic state space associated with the environment.
        Could be empty in environments that only have a raw state.
        """
        return self._oracle_space

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    def current_oracle_state(self) -> np.ndarray:
        """
        Returns the oracle state as a raw value; i.e., not a Dict[Variable, Expression].
        The `Env.oracle_space` can be used to convert this into something with names.
        """
        return self._state.copy()

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError
