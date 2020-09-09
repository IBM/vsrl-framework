#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import random
from math import cos, isnan, pi, sin, sqrt
from typing import Dict, Optional, Tuple

import numpy as np
import portion
import torch
from PIL import Image

import vsrl.verifier.expr as vexpr
from vsrl.rl.envs.render_helpers import paste_coordinates
from vsrl.spaces import CompactSet
from vsrl.symmap.symbolic_mapper import SymFeatExtractor
from vsrl.utils.assets import get_image_path

from ._utils import EpisodeTerminatedException, gen_separated_points
from .env import Env

REACHED_STEP_LIMIT = 4
MOVED_OFF_MAP = 3
REACHED_GOAL_HAZARD = 2
REACHED_GOAL = 1
NOT_DONE = 0
REACHED_HAZARD = -1


class PMGFSymFeatExtractor(SymFeatExtractor):
    agent_id = 0
    hazard_id = 2

    # TODO: make it easier for max_n_obstacles to stay in sync with PMGF
    def __init__(self, detector: torch.nn.Module, max_n_obstacles: int = 10):
        super().__init__(detector)
        self.max_n_obstacles = max_n_obstacles
        self._range = torch.nn.Parameter(
            torch.arange(
                PMGoalFinding._obs_start_idx,
                PMGoalFinding._obs_start_idx + 2 * max_n_obstacles,
                2,
            ),
            requires_grad=False,
        )

    def forward(self, imgs):
        """
        The symbolic state vector has the following data in it at the given indices:
        [0, 1]: ego_x, ego_y
        [2, 3]: goal_x, goal_y (nan; we do detect this but it isn't used)
        [4]: theta (nan but we should try to estimate this at some point?)
        [5, 6]: v, w (nan)
        [7:]: hazard1_x, hazard1_y, hazard2_x, ...

        If a detection for `ego` is missing, (0, 0) will be used. For hazards, because
        the number could be variable, (nan, nan) is used if the number is less than
        max_n_obstacles.

        :param imgs: nchw
        :returns: n x d where d = 7 + 2 * max_n_obstacles
        """
        img_idx, class_id, center_x, center_y, _ = super().forward(imgs)
        sym_feats = torch.full(
            (len(imgs), 7 + 2 * self.max_n_obstacles), float("nan"), device=imgs.device
        )

        agent_idx = class_id == self.agent_id
        agent_img_idx = img_idx[agent_idx]
        hazard_idx = class_id == self.hazard_id
        hazard_img_idx = img_idx[hazard_idx]

        sym_feats[:, :2] = 0  # ensure no NaNs for agent location
        sym_feats[agent_img_idx, 0] = center_x[agent_idx]
        sym_feats[agent_img_idx, 1] = center_y[agent_idx]

        _, counts = torch.unique(hazard_img_idx, return_counts=True)
        col_idx = torch.cat([self._range[:c] for c in counts])
        sym_feats[hazard_img_idx, col_idx] = center_x[hazard_idx]
        sym_feats[hazard_img_idx, col_idx + 1] = center_y[hazard_idx]
        return sym_feats


class PMGoalFinding(Env):
    """
    DSolve[odes = {
        x'[t] == v[t]*Cos[theta[t]],
        y'[t] == v[t]*Sin[theta[t]],
        v'[t] == a,
        theta'[t] == w,
        theta[0] == theta0,
        x[0] == x0,
        y[0] == y0,
        v[0] == v0
      }, {x[t], y[t], v[t], theta[t]}, t]

    theta[t] -> theta0 + t w
    v[t] -> a t + v0,
    x[t] -> (1/(w^2))(w^2 x0 - a Cos[theta0] -
      v0 w Sin[theta0] + a Cos[theta0 + t w] +
      a t w Sin[theta0 + t w] + v0 w Sin[theta0 + t w]),
    y[t] -> (1/(w^2))(w^2 y0 + v0 w Cos[theta0] -
      a Sin[theta0] - a t w Cos[theta0 + t w] -
      v0 w Cos[theta0 + t w] + a Sin[theta0 + t w])
    """

    SymFeatClass = PMGFSymFeatExtractor
    # indices into self._state for certain parts of the state
    _ego_x_idx: int = 0
    _ego_y_idx: int = 1
    _goal_x_idx: int = 2
    _goal_y_idx: int = 3
    _theta_idx: int = 4
    _sin_theta_idx = 5
    _cos_theta_idx = 6
    _v_idx: int = 7
    _w_idx: int = 8
    _obs_start_idx: int = 9  # index of first obstacle. from here on, the state is
    # (obs0_x, obs0_y, obs1_x, ...) for all num_obstacles obstacles

    def __init__(
        self,
        num_obstacles: int = 10,  # TODO
        img_scale: int = 1,
        grayscale: bool = False,
        oracle_obs: bool = False,
        safe_sep: float = 1.0,
        extra_hazard_size: Optional[int] = None,
        walls: bool = False,
        dense_rewards: bool = False,
        randomize_goal: bool = False,
    ):
        """
        :param extra_hazard_size: for debugging only; this renders hazards a second time
          with half alpha and with height and width increased by this amount. This is
          useful to visualize how far a safe agent has to stay away from hazards if it
          has a certain number of pixels of maximum detection error.
        :param walls: don't let the agent go off the screen; no boundary penalty
        :param dense_rewards: rewards for distance + angle to goal
        """
        scene = Image.open(get_image_path("top/bg.png"))
        ego_img = Image.open(get_image_path("top/blue.png"))
        goal_img = Image.open(get_image_path("top/goal.png"))
        hazard_img = Image.open(get_image_path("top/hazard.png"))
        hazard_x_idx = [self._obs_start_idx + 2 * i for i in range(num_obstacles)]
        hazard_y_idx = [x_idx + 1 for x_idx in hazard_x_idx]
        objs = {
            "ego": (ego_img, self._ego_x_idx, self._ego_y_idx, self._theta_idx),
            "goal": (goal_img, self._goal_x_idx, self._goal_y_idx),
            "hazard": (hazard_img, hazard_x_idx, hazard_y_idx),
        }

        # load graphics
        scene = Image.open(get_image_path("top/bg.png"))

        self.map_buffer = 12
        self.T = 0.1
        self.A = 30  # can change v by max 3 per step
        self.B = 30  # should be positive; accel range is [-B, A]
        self.min_w = -1
        self.max_w = 1
        self.max_v = 30  # max 3 pixels per step
        self._safe_sep = safe_sep
        self.num_obstacles = num_obstacles
        self._extra_hazard_size = extra_hazard_size
        self._walls = walls
        self._dense_rewards = dense_rewards
        self._randomize_goal = randomize_goal
        self.place_pointers = False

        # [v, cos(theta), sin(theta)]
        vector_obs_bounds = ([0, -1, -1], [self.max_v, 1, 1])
        super().__init__(
            img_scale,
            grayscale,
            oracle_obs,
            scene,
            objs,
            vector_obs_bounds=vector_obs_bounds,
        )

        # _radius determines if a collision is happening. We'll use the sizes of the
        # images to directly compute when they overlap.
        assert hazard_img.size[0] == hazard_img.size[1]
        # not true for ego img, but width is larger, so radius is still okay
        # assert self._egoimg.size[0] == self._egoimg.size[1]
        assert goal_img.size[0] == hazard_img.size[0]
        self._radius = ego_img.size[0] / (2 * img_scale) + hazard_img.size[0] / (
            2 * img_scale
        )
        self._safe_sep += self._radius
        self._max_goal_dist = sqrt(self._width ** 2 + self._height ** 2)

    def _make_action_space(self) -> CompactSet:
        return CompactSet(
            {
                vexpr.Variable("w"): (self.min_w, self.max_w),  # rotational velocity
                vexpr.Variable("a"): (-self.B, self.A),  # translational acceleration
            }
        )

    def _make_oracle_space(self) -> CompactSet:
        # make sure this stays in sync with the indices into the state array defined
        # in init
        ranges: Dict[vexpr.Variable, Tuple[float, float]] = {
            vexpr.Variable("ego_x"): (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            ),
            vexpr.Variable("ego_y"): (
                0 + self.map_buffer,
                self._height - self.map_buffer,
            ),
            vexpr.Variable("goal_x"): (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            ),
            vexpr.Variable("goal_y"): (
                0 + self.map_buffer,
                self._height - self.map_buffer,
            ),
            vexpr.Variable("theta"): (-2 * pi, 2 * pi),
            vexpr.Variable("sin_theta"): (-1, 1),
            vexpr.Variable("cos_theta"): (-1, 1),
            vexpr.Variable("v"): (0, self.max_v),
            vexpr.Variable("w"): (self.min_w, self.max_w),
        }
        # Add obstacles to the oracle space.
        for i in range(1, self.num_obstacles + 1):
            ranges[vexpr.Variable(f"obs{i}_x")] = (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            )
            ranges[vexpr.Variable(f"obs{i}_y")] = (
                0 + self.map_buffer,
                self._height - self.map_buffer,
            )
        return CompactSet(ranges)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        :param action: w, a
        """
        if self._done:
            raise EpisodeTerminatedException()
        assert action in self.action_space
        w, a = action

        v0 = self._state[self._v_idx]
        new_v = v0 + a * self.T
        # enforce the bounds on v ([0, 1]) by changing a.
        if new_v < 0:
            a = -v0 / self.T
            new_v = 0
        elif new_v > self.max_v:
            a = (self.max_v - v0) / self.T
            new_v = self.max_v

        theta0 = self._state[self._theta_idx]
        sin_theta0 = self._state[self._sin_theta_idx]
        cos_theta0 = self._state[self._cos_theta_idx]
        new_state = self._state.copy()

        tw = self.T * w
        theta = theta0 + tw
        # keep theta in [-2 * pi, 2 * pi]
        if theta > 2 * pi:
            theta -= 2 * pi
        elif theta < -2 * pi:
            theta += 2 * pi

        cos_theta = cos(theta)
        sin_theta = sin(theta)

        new_state[self._w_idx] = w
        new_state[self._v_idx] += a * self.T
        new_state[self._theta_idx] = theta
        new_state[self._sin_theta_idx] = sin_theta
        new_state[self._cos_theta_idx] = cos_theta

        if w != 0:
            new_state[self._ego_x_idx] += (1 / w ** 2) * (
                -a * cos_theta0
                - v0 * w * sin_theta0
                + a * cos_theta
                + a * tw * sin_theta
                + v0 * w * sin_theta
            )
            new_state[self._ego_y_idx] += (1 / w ** 2) * (
                v0 * w * cos_theta0
                - a * sin_theta0
                - a * tw * cos_theta
                - v0 * w * cos_theta
                + a * sin_theta
            )
        else:
            new_state[self._ego_x_idx] += cos_theta0 * max(
                0, v0 * self.T + a * self.T ** 2 / 2
            )
            new_state[self._ego_y_idx] += sin_theta0 * max(
                0, v0 * self.T + a * self.T ** 2 / 2
            )

        if self._walls:  # keep the agent in-bounds; no boundary penalties
            x = new_state[self._ego_x_idx]
            y = new_state[self._ego_y_idx]
            new_state[self._ego_x_idx] = min(
                max(self.map_buffer, x), self._width - self.map_buffer
            )
            new_state[self._ego_y_idx] = min(
                max(self.map_buffer, y), self._height - self.map_buffer
            )

        self._step += 1
        reward, done_code = self._get_reward_and_done_code(
            self._state, action, new_state
        )

        self._state = new_state
        self._done = done_code != NOT_DONE
        obs = self._get_obs(np.array([new_v, cos_theta, sin_theta], dtype=np.float32))

        return (
            obs,
            reward,
            self._done,
            {
                "done_reason": done_code,
                "unsafe": done_code in (REACHED_GOAL_HAZARD, REACHED_HAZARD),
            },
        )

    def _is_valid_state(self, state):
        return state in self.oracle_space

    def _final_state_description(self, done):
        if done == MOVED_OFF_MAP:
            return "moved off map"
        elif done == REACHED_GOAL_HAZARD:
            return "reached goal and hazard at the same time"
        elif done == REACHED_GOAL:
            return "reached goal and not currently touching any hazards"
        elif done == NOT_DONE:
            return "not done"
        elif done == REACHED_HAZARD:
            return (
                f"crashed into a hazard at position {self._find_collision(self._state)}"
            )
        elif done == REACHED_STEP_LIMIT:
            return f"Reached limit of {self.horizon:,} steps."
        else:
            return f"done, but not sure why. Done code was {done}"

    def _get_reward_and_done_code(self, state1, action, state2) -> Tuple[float, int]:
        """
        :param state:
        :return: (reward, done_code). Use _final_state_description to interpret done_code.
        """
        egox, egoy = state2[:2]
        vector_to_goal = state2[2:4] - state2[:2]
        dist_to_goal = sqrt(np.dot(vector_to_goal, vector_to_goal))
        # if we have gamma = 0.99 then getting a constant reward c for H steps gives
        # reward < 100c. We want it to be better to go to the goal right away than to
        # get the distance reward from right next to the goal, so we make its maximum
        # value < 1 / 100 * goal reward.
        # or, to be careful, in case gamma = 1, we can make
        # max_dist_reward * H < goal_reward
        if self._dense_rewards:
            vector_to_goal /= dist_to_goal  # normalize
            theta = state2[self._theta_idx]
            angle_to_goal = np.dot(vector_to_goal, np.array([cos(theta), sin(theta)]))
            # convert both rewards to [0, 1] then scale their sum
            dist_reward = 1 - dist_to_goal / self._max_goal_dist
            angle_reward = angle_to_goal / 2 + 0.5
            reward = (dist_reward + angle_reward) / 20
        else:
            reward = 0

        at_goal = dist_to_goal <= self._radius
        is_colliding = self._collision(state2)
        if not (
            self.map_buffer <= egox <= self._width - self.map_buffer
            and self.map_buffer <= egoy <= self._height - self.map_buffer
        ):
            return -1, MOVED_OFF_MAP
        if at_goal:
            if is_colliding:
                return 0, REACHED_GOAL_HAZARD
            return 10, REACHED_GOAL
        if is_colliding:
            return -1, REACHED_HAZARD
        if self._step >= self.horizon:
            return reward, REACHED_STEP_LIMIT
        return reward, NOT_DONE

    def _collision(self, state):
        return self._find_collision(state) is not None

    def _find_collision(self, state) -> Optional[Tuple[int, int]]:
        egox = state[self._ego_x_idx]
        egoy = state[self._ego_y_idx]
        for i in range(self.num_obstacles):
            xidx = self._obs_start_idx + 2 * i
            yidx = xidx + 1
            ox, oy = state[xidx], state[yidx]
            if self._circle_contains(egox, egoy, ox, oy, self._radius):
                return ox, oy
        return None

    def _circle_contains(self, x, y, c_x, c_y, c_radius):
        return (x - c_x) ** 2 + (y - c_y) ** 2 <= c_radius ** 2

    def reset(self) -> np.ndarray:
        state = np.empty_like(self._state)
        angle = random.random() * 2 * pi

        if self._randomize_goal:
            initial_points = None
        else:
            initial_points = np.array([[3 * self._width // 4, 3 * self._height // 4]])

        # place objects so they don't collide
        points = gen_separated_points(
            self.num_obstacles + 2,
            sep=self._radius,
            lower_bounds=np.array([self.map_buffer, self.map_buffer]),
            upper_bounds=np.array(
                [self._width - self.map_buffer, self._height - self.map_buffer]
            ),
            initial_points=initial_points,
        )
        state[self._w_idx] = 0
        state[self._v_idx] = 0
        state[self._theta_idx] = angle
        state[self._sin_theta_idx] = sin(angle)
        state[self._cos_theta_idx] = cos(angle)
        state[self._goal_x_idx] = points[0, 0]
        state[self._goal_y_idx] = points[0, 1]
        state[self._ego_x_idx] = points[1, 0]
        state[self._ego_y_idx] = points[1, 1]
        state[self._obs_start_idx :] = points[2:].ravel()

        assert self._is_valid_state(state), self.oracle_space.to_state(state)
        self._state = state
        self._done = False
        self._step = 0
        self._prev_frame.fill(0)
        obs = self._get_obs(np.array([0, cos(angle), sin(angle)], dtype=np.float32))
        return obs

    def state_constants(self):
        return {
            vexpr.Variable("A"): vexpr.Number(self.A),
            vexpr.Variable("B"): vexpr.Number(self.B),
            vexpr.Variable("T"): vexpr.Number(self.T),
            vexpr.Variable("safe_sep"): vexpr.Number(self._safe_sep),
            vexpr.Variable("min_w"): vexpr.Number(self.min_w),
            vexpr.Variable("max_w"): vexpr.Number(self.max_w),
        }

    @staticmethod
    def constraint_func(action, sym_feats, B, T, safe_sep):
        """
        L-inf norm version: (doesn't take direction of the agent into account)
        (
            abs(x - ox) > v^2 / (2 * B) + (A / B + 1) * (A / 2 * T^2 + T * v)
            | abs(y - oy) > v^2 / (2 * B) + (A / B + 1) * (A / 2 * T^2 + T * v)
        )
        (`A` here is the current desired acceleration, not the max acceleration)

        Note that the environment bounds the velocity to [0, max_v] by having the
        acceleration action not set the acceleration directly if the velocity would go
        out of bounds. The constraint doesn't take this into account, so some
        acceleration actions might be deemed unsafe which actually would be safe to take
        (but the actual acceleration used in the environment step would be different than
        the action in such cases).

        The distance in the constraint is equivalent to:
        pos_diff = v * T + a / 2 * T ** 2  # after taking one step with accel of `a`
        v_new = v + a * T
        stop_time = v_new / B  # stop time / distance after one step
        stop_dist = v_new * stop_time + -B / 2 * stop_time ** 2
        safe_dist = pos_diff + stop_dist

        WARNING: pos_diff could be negative here, but we don't allow negative v.
        If this presents an issue, we should modify the constraint to take into account
        how the environment changes `a` if `v` would become negative (or change the
        environment to make `T` smaller in such cases instead of changing `a`. I worry
        that might make the learning more difficult, though, especially if the agent
        repeatedly tries to use a very negative `a` when `v` is already nearly 0.)
        """
        w, a = action
        x, y = sym_feats[:2]
        v = sym_feats[PMGoalFinding._v_idx]

        safe_dist = (
            (v ** 2 / (2 * B)) + ((a / B + 1) * (a / 2 * T ** 2 + T * v))
        ) + safe_sep

        for i in range(PMGoalFinding._obs_start_idx, len(sym_feats), 2):
            ox = sym_feats[i]
            if isnan(ox):  # once one object is nan, all others will be
                break
            oy = sym_feats[i + 1]
            if abs(x - ox) < safe_dist and abs(y - oy) < safe_dist:
                return False
        return True

    @staticmethod
    def constrained_sample(sym_feats, min_w, max_w, B, A, T, safe_sep):
        """
        w is unconstrained (at least with the l-inf norm constraint; if we use a less
        conservative constraint, we'll have to deal with w too)

        The sampling here is more complex because the possible values for acc are the
        intersection of possible values for each hazard's constraint and each hazard has
        a union of two terms as a constraint. There will only ever be two compact ranges
        that contain the safe space for all objects considered so far, but we have to
        track these carefully.

        Start with the constraints (shown for x but almost identical for y):
          abs(x - ox) > v^2 / (2 * B) + (A / B + 1) * (A / 2 * T^2 + T * v)
        then write as a quadratic in A:
          0 > A^2 * a + A * b + (c - abs(x - ox))
          a = T^2 / (2 * B)
          b = T * V / B + T^2 / 2
          c = T * v + v^2 / (2 * B)
        Use the quadratic formula to find the zeros (only abs(x - ox) has to change for
        each object and for x / y). a > 0, so the area between the zeros is the safe set
        for A (unioned between x and y, intersected over all hazards).
        """
        w = min_w + (max_w - min_w) * random.random()

        x, y = sym_feats[:2]
        v = sym_feats[PMGoalFinding._v_idx]

        a = T ** 2 / (2 * B)
        b = T * v / B + T ** 2 / 2
        c0 = T * v + v ** 2 / (2 * B) + safe_sep

        safe_sets = portion.closed(-B, A)
        for i in range(PMGoalFinding._obs_start_idx, len(sym_feats), 2):
            ox = sym_feats[i]
            if isnan(ox):
                break
            oy = sym_feats[i + 1]

            c_x = c0 - abs(x - ox)
            c_y = c0 - abs(y - oy)

            d = b ** 2 - 4 * a * c_x
            if d < 0:  # fallback action is -B
                z11, z12 = -B, -B
            else:
                d = sqrt(d)
                z12 = (-b + d) / (2 * a)
                z11 = (-b - d) / (2 * a)
                z11, z12 = (z11, z12) if z11 < z12 else (z12, z11)

            d = b ** 2 - 4 * a * c_y
            if d < 0:
                z21, z22 = -B, -B
            else:
                d = sqrt(d)
                z22 = (-b + d) / (2 * a)
                z21 = (-b - d) / (2 * a)
                z21, z22 = (z21, z22) if z21 < z22 else (z22, z21)

            safe_sets &= portion.closed(z11, z12) | portion.closed(z21, z22)

        if safe_sets.empty:
            return np.array([max_w, -B], dtype=np.float32)
        volumes = tuple(s.upper - s.lower for s in safe_sets)
        safe_set = random.choices(safe_sets, weights=volumes)[0]
        acc = safe_set.lower + (safe_set.upper - safe_set.lower) * random.random()
        return np.array([w, acc], dtype=np.float32)
