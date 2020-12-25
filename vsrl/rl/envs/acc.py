#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import random
from math import sqrt
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from vsrl.spaces.continuous import CompactSet
from vsrl.spaces.fin_space import FiniteSpace
from vsrl.spaces.space import Space
from vsrl.symmap.symbolic_mapper import SymFeatExtractor
from vsrl.utils.assets import get_image_path
from vsrl.verifier import expr as vexpr

from ._utils import EpisodeTerminatedException
from .env import Env

# Note: this is the safety constraint for a modified environment where the lead agent can move:
# (max((v+a*T,v))-min((0,ov-B*T)))^2+2*(B-A)*(ox+ov*T+B*T^2/2-(x+v*T+a*T^2/2)+S) < 0


class ACCSymFeatExtractor(SymFeatExtractor):
    def forward(self, imgs):
        img_idx, obj_id, center_x, _, _ = super().forward(imgs)
        # if a detection is missing, a value of 0 will be used. If there are multiple
        # detections for the same object type in one image, the last one is used

        # detector has leader: 0, follower: 1
        # sym features need follower: 0, rel velocity: 1, leader: 2
        obj_id[obj_id == 0] = 2
        obj_id[obj_id == 1] = 0
        sym_feats = torch.zeros(len(imgs), 3, device=imgs.device)
        sym_feats[img_idx, obj_id] = center_x
        sym_feats[:, 1] = float("nan")
        return sym_feats  # n_imgs x (follower_pos, rvel (nan), leader_pos)


class ACC(Env):

    SymFeatClass = ACCSymFeatExtractor
    _ego_x_idx = 0
    _rel_vel_idx = 1
    _leader_x_idx = 2

    def __init__(
        self,
        continuous_action_space: bool = True,
        A: int = 5,
        B: int = 5,
        T: float = 0.5,
        safe_sep: float = 1.0,
        img_scale: int = 1,
        grayscale: bool = False,
        oracle_obs: bool = False,
        walls: bool = False,
        dense_rewards: bool = False,
    ):
        """
        :param continuous_action_space: if True the agent's action space is [-B, A];
          otherwise, it's {-B, A}.
        """
        assert A > 0 and B > 0 and T > 0

        scene = Image.open(get_image_path("lateral/bg.png"))
        ego_img = Image.open(get_image_path("lateral/blue.png"))
        leader_img = Image.open(get_image_path("lateral/red.png"))
        self._y_coord = 3 / 5 * scene.height / img_scale + 37 / img_scale
        objs = {
            "ego": (ego_img, self._ego_x_idx, self._y_coord),
            "leader": (leader_img, self._leader_x_idx, self._y_coord),
        }

        self.continuous_action_space = continuous_action_space

        self.A = A
        self.B = B
        self.T = T
        self._safe_sep = safe_sep
        self._buffer_space = 1
        self.show_pointers = False

        # [relative velocity]
        vector_obs_bounds = ([-100], [100])  # TODO: these bounds aren't enforced
        super().__init__(
            img_scale,
            grayscale,
            oracle_obs,
            scene,
            objs,
            vector_obs_bounds=vector_obs_bounds,
        )

        self.action_space = self._make_action_space()


        # graphics
        self._init_state = np.array(
            [
                self._width / 4,  # car position
                0,  # the relative velocity of the two cars.
                3 * self._width / 4,  # leader position
            ],
            dtype=np.float32,
        )

        self._safe_sep += (ego_img.width + leader_img.width) / (2 * img_scale)
        # the reward-optimal distance back from the leader
        self._optimal_dist = self._safe_sep + ego_img.width / img_scale

    def reset(self, initial_state: np.ndarray = None) -> np.ndarray:
        if initial_state is None:
            initial_state = self._init_state.copy()
            action = np.array([0], dtype=np.float32)
            max_x = initial_state[2] - self._safe_sep
            while True:
                initial_state[0] = random.random() * max_x
                initial_state[1] = -self.B + (self.A + self.B) * random.random()
                if self.constraint_func(action, initial_state):
                    break
        self._state = initial_state
        self._done = False
        self._step = 0
        self._prev_frame.fill(0)
        return self._get_obs(self._state[1:2])

    def _make_oracle_space(self) -> Space:
        return CompactSet(
            {
                vexpr.Variable("follower_pos"): (0, self._width),
                vexpr.Variable("rvel"): (-100, 100),
                vexpr.Variable("leader_pos"): (0, self._width),
            }
        )

    def _make_action_space(self) -> Space:
        if self.continuous_action_space:
            return CompactSet({vexpr.Variable("acc"): (-self.B, self.A)})
        else:
            accel_action = np.array([self.A])
            decel_action = np.array([-self.B])
            return FiniteSpace([accel_action, decel_action], [vexpr.Variable("acc")])

    def step(self, action: np.ndarray) -> Tuple[object, float, bool, dict]:
        assert action in self.action_space, f"{action} was not in the action space."
        action = action[0]
        if self._done:
            raise EpisodeTerminatedException()
        self._step += 1
        x, rv, ox = self._state
        new_state = np.empty(3, dtype=np.float32)
        # min(:, ox - 5) instead of min(:, _width) is a hack to deal with collisions
        new_state[0] = min(max(x + rv * self.T + (action * self.T ** 2) / 2, 0), ox - 5)
        new_state[1] = rv + action * self.T
        new_state[2] = ox
        reward = self.compute_reward(self._state, action, new_state)
        self._state = new_state
        rv = (
            self._get_obs(self._state[1:2]),
            reward,
            self._is_done(),
            {"unsafe": self._is_crashed(self._state)},
        )
        return rv

    def _is_crashed(self, s):
        return s[0] > s[2] - self._safe_sep

    def compute_reward(self, s1, a, s2):
        if s2[0] <= 0:
            return -10
        elif self._is_crashed(s2):
            return -10
        else:
            optimal_x = s2[2] - self._optimal_dist
            # this makes the reward in [0, 1]
            return 1 - abs(s2[0] - optimal_x) / self._width

    def _is_done(self):
        x = self._state[0]
        self._done = (
            x <= 0 or self._is_crashed(self._state) or self._step >= self.horizon
        )
        return self._done

    def constraint_func(self, action, sym_feats):
        """

        (could make this an instance method and do self.B, self.T, self.SAFE_STEP
        instead of using partial in RLPytEnv to pass these in)

        This isn't quite accurate - if you have v > 0 (moving right) then set acc < 0 and
        end with v_new <= 0, this says you're safe. But it's possible you hit the leader
        before you started moving left (this is more of an issue the larger T is).
        I think this is okay because the environment itself doesn't actually check for such
        collisions.

        If we prefer a string version (for proving correctness and reducing bugs in
        translation from the proven constraint):

        (
        (acc = -B)
        | (v + acc * T <= 0 & x_f + v * T + 1 / 2 * acc * T ** 2 + safe_sep <= x_l)
        | (v + acc * T > 0 & x_f + v * T + 1 / 2 * acc * T ** 2 + (v + acc * T) ** 2 / (2 * B) + safe_sep <= x_l)
        )
        (the variables need renaming here to match ACC's names)
        """
        (acc,) = action
        x_f, v, x_l = sym_feats
        if acc == -self.B:
            return True
        x_new = x_f + v * self.T + 1 / 2 * acc * self.T ** 2
        v_new = v + acc * self.T
        if v_new <= 0:
            return x_new + self._safe_sep <= x_l
        stop_time = v_new / self.B
        stop_dist = v_new * stop_time + 1 / 2 * (-self.B) * stop_time ** 2
        return x_new + stop_dist + self._safe_sep <= x_l

    def constrained_sample(self, sym_feats):
        """
        If all sets have no volume, the last one is always chosen; this is a bug but it doesn't
        seem likely to matter right now. (this seems to be the behavior of random.choices with
        all weights = 0). You could add some eps to each volume to fix this or just do random.choice
        if all volumes are 0.
        """
        x_f, v, x_l = sym_feats

        a = self.T ** 2 / (2 * self.B)
        b = 1 / 2 * self.T ** 2 + v * self.T / self.B
        c = x_f + self._safe_sep + v * self.T + v ** 2 / (2 * self.B) - x_l

        d = b ** 2 - 4 * a * c
        if d < 0:
            z1, z2 = -self.B, -self.B
        else:
            d = sqrt(d)
            z2 = (-b + d) / (2 * a)
            z1 = (-b - d) / (2 * a)
            z1, z2 = (z1, z2) if z1 < z2 else (z2, z1)
            # this constraint is convex (a > 0), so to be <= 0 we need the area
            # between the zero crossings

        # (lb, ub); note that [-B, A] are always bounds
        sets = (
            (-self.B, -self.B),
            (
                -self.B,
                min(
                    -v / self.T,
                    2 * (x_l - x_f - v * self.T - self._safe_sep) / self.T ** 2,
                    self.A,
                ),
            ),
            (max(z1, -self.B), min(z2, self.A)),
        )
        volumes = tuple(ub - lb for lb, ub in sets)
        lb, ub = random.choices(sets, weights=volumes)[0]
        # eps = 1e-8
        # ub = max(lb, ub - eps)
        action = lb + (ub - lb) * random.random()
        # the max is to prevent some numerical issues when -B = lb ~= ub
        return np.array([max(action, -self.B)], dtype=np.float32)
