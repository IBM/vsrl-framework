#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from abc import ABC, abstractmethod
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import gym
import numpy as np
from PIL import Image
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox

from vsrl.spaces.space import Space


class Observation(NamedTuple):
    img: np.ndarray
    vector: np.ndarray  # any non-image (1D) features (velocity, angle)


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
        scene: Optional[Image.Image] = None,
        preprocess_img_names: Sequence[str] = (),
        height: Optional[int] = None,
        width: Optional[int] = None,
        horizon: int = 1000,
        vector_obs_bounds: Optional[Tuple[List[int], List[int]]] = None,
    ):
        """
        :param img_scale: how much to scale the images (background + objects) down by.
          If 1, no scaling is used. If, e.g., 2, the images are half their usual size.
        :param graycsale: if True, all images (and thus observations) are converted to
          grayscale.
        :param oracle_obs: whether to return oracle observations (if True) or images
        :param preprocess_img_names: the names of the images which will be downscaled /
          grayscaled (these should all be attributes of `self` already except _scene
          which is assigned here)
        :param height: `height` and `width` are extracted from `scene`; they should only
          be given explicitly if `scene` is None
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
        self._scene = scene
        self.horizon = horizon

        self._preprocess_imgs(preprocess_img_names)
        if scene is not None:
            self._width, self._height = self._scene.size
        else:
            if height is None or width is None:
                raise ValueError("height and width must be given if scene is not.")
            self._height = height
            self._width = width
        c = 1 if grayscale else 3
        self._prev_frame = np.zeros((self._height, self._width, c), dtype=np.uint8)
        self._oracle_space = self._make_oracle_space()
        self._action_space = self._make_action_space()
        self.action_space = gym.spaces.Box(
            self._action_space.lower_bounds, self._action_space.upper_bounds
        )
        # this state doesn't have to be entirely correct; reset() is called before step
        self._state = self.oracle_space.sample().astype(np.float32)
        self._done = True
        self._step = 0

        if oracle_obs:
            self.observation_space = gym.spaces.Box(
                self.oracle_space.lower_bounds, self.oracle_space.upper_bounds
            )
        else:
            gym.spaces.Dict({"img": gym.spaces.Box(0, 255, shape=(10, 10, 3))})
            img_obs_space = gym.spaces.Box(
                0, 255, shape=(self._height, self._width, 2 * c)
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

    def _preprocess_imgs(self, img_names: Iterable[str]) -> None:
        for img_name in img_names:
            img = getattr(self, img_name)
            img.load()
            if self._img_scale > 1:
                img = img.reduce(self._img_scale)
            mask = img.split()[-1]  # alpha channel
            if self._grayscale:
                img = img.convert("L")  # "luminosity"
            else:
                img = img.convert("RGB")
            if img_name != "_scene":
                img.mask = mask

            setattr(self, img_name, img)

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

    @abstractmethod
    def render(self) -> Union[Image.Image, np.ndarray]:
        raise NotImplementedError

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
