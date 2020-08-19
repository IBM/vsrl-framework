#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import random
from math import cos, isnan, pi, sin, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import portion
import torch
from PIL import Image

import vsrl.verifier.expr as vexpr
from vsrl.rl.envs.render_helpers import paste_coordinates
from vsrl.spaces.continuous import CompactSet
from vsrl.spaces.space import Space
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

# todo:
# 1. implement logic for removing a pointmess using at_pointmesses
# 2. give reward for being at a pointmess
# 3. implement logic for add pointmesses using _collision
# 3. test

# TODO: make PMGF a subclass of this which just always sets the number of
# pointmesses (initial + on collision) to 0. Some changes need to be made
# (e.g. in the reward / done check) to make this work with no pointmesses
# on collision and, if desired, with the dense rewards for PMGF. Also
# to make the sym feat extractor work for either (it just needs to know
# the number of spaces to leave for pointmesses)


class PMSymFeatExtractor(SymFeatExtractor):
    agent_id = 0
    hazard_id = 2

    # TODO: make it easier for max_n_obstacles to stay in sync with PM
    def __init__(
        self,
        detector: torch.nn.Module,
        output_scale: int,
        max_n_obstacles: int = 10,
        num_initial_pointmesses: int = 10,
        num_pointmesses_on_collision: int = 3,
    ):
        super().__init__(detector, output_scale)
        self._max_n_obstacles = max_n_obstacles
        self._num_initial_pointmesses = num_initial_pointmesses
        self._num_pointmesses_on_collision = num_pointmesses_on_collision
        self._range = torch.nn.Parameter(
            torch.arange(
                Pointmesses._obs_start_idx,
                Pointmesses._obs_start_idx + 2 * max_n_obstacles,
                2,
            ),
            requires_grad=False,
        )
        self._n_sym_feats = 7 + 2 * (
            max_n_obstacles * (num_pointmesses_on_collision + 1)
            + num_initial_pointmesses
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
            (len(imgs), self._n_sym_feats), float("nan"), device=imgs.device
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


class Pointmesses(Env):

    SymFeatClass = PMSymFeatExtractor

    # indices into self._state for certain parts of the state
    _ego_x_idx: int = 0
    _ego_y_idx: int = 1
    _goal_x_idx: int = 2
    _goal_y_idx: int = 3
    _theta_idx: int = 4
    _v_idx: int = 5
    _w_idx: int = 6
    _obs_start_idx: int = 7  # index of first obstacle. from here, the state is
    # (obs0_x, obs0_y, obs1_x, ...) for num_obstacles obstacles then
    # (pm0_x, pm0_y, pm1_x, ...) for all pointmesses

    def __init__(
        self,
        num_obstacles: int = 10,
        num_initial_pointmesses: int = 10,
        num_pointmesses_on_collision: int = 3,
        img_scale: int = 1,
        grayscale: bool = False,
        oracle_obs: bool = False,
        safe_sep: float = 1.0,
        walls: bool = False,
        dense_rewards: bool = False,
    ):
        scene = Image.open(get_image_path("top/bg.png"))
        ego_img = Image.open(get_image_path("top/blue.png"))
        goal_img = Image.open(get_image_path("top/goal.png"))
        hazard_img = Image.open(get_image_path("top/hazard.png"))
        pm_img = Image.open(get_image_path("top/pointmess.png"))
        hazard_x_idx = [self._obs_start_idx + 2 * i for i in range(num_obstacles)]
        hazard_y_idx = [x_idx + 1 for x_idx in hazard_x_idx]
        n_pm_total = (
            num_initial_pointmesses + num_pointmesses_on_collision * num_obstacles
        )
        pm_x_idx = [
            self._obs_start_idx + num_obstacles * 2 + 2 * i for i in range(n_pm_total)
        ]
        pm_y_idx = [x_idx + 1 for x_idx in pm_x_idx]
        objs = {
            "ego": (ego_img, self._ego_x_idx, self._ego_y_idx),
            "goal": (goal_img, self._goal_x_idx, self._goal_y_idx),
            "hazard": (hazard_img, hazard_x_idx, hazard_y_idx),
            "pointmess": (pm_img, pm_x_idx, pm_y_idx),
        }

        self.num_obstacles = num_obstacles
        self.num_initial_pointmess = num_initial_pointmesses
        self.num_pointmesses_on_collision = num_pointmesses_on_collision
        # properties of the map
        self.map_buffer = 12
        # properties of the dynamics
        self.T = 0.1
        self.A = 30
        self.B = 30
        self.min_w = -1
        self.max_w = 1
        self.max_v = 30
        self._safe_sep = safe_sep
        # configuration variables for debugging.
        # If true, we'll place yellow dots at the midpoints of various objects.
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

        # properties relevant to computing collisions.
        # RADIUS determines if a collision is happening. We'll use the sizes of the images to directly compute when they
        # overlap.
        assert hazard_img.size[0] == hazard_img.size[1]
        assert goal_img.size[0] == hazard_img.size[0]
        assert pm_img.size[0] == pm_img.size[1]
        self._hazard_radius = (ego_img.size[0] + hazard_img.size[0]) / (2 * img_scale)
        self._safe_sep += self._hazard_radius
        self._pointmess_radius = (ego_img.size[0] + pm_img.size[0]) / (2 * img_scale)

    def _make_action_space(self) -> Space:
        return CompactSet(
            {
                vexpr.Variable("w"): (self.min_w, self.max_w),  # rotational velocity
                vexpr.Variable("a"): (-self.B, self.A),  # translational acceleration
            }
        )

    def _make_oracle_space(self) -> Space:
        """Contains an entry for the ego, the charger, every obstacle, initial pointmesses, and `self.num_pointmesses_on_collision` pointmesses per obstacle"""
        ranges: Dict[vexpr.Variable, Tuple[float, float]] = {
            vexpr.Variable("ego_x"): (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            ),
            vexpr.Variable("ego_y"): (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            ),
            vexpr.Variable("charger_x"): (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            ),
            vexpr.Variable("charger_y"): (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            ),
            vexpr.Variable("theta"): (-2 * pi, 2 * pi),
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
        for i in range(
            1,
            self.num_initial_pointmess
            + self.num_pointmesses_on_collision * self.num_obstacles
            + 1,
        ):
            ranges[vexpr.Variable(f"pointmess{i}_x")] = (
                0 + self.map_buffer,
                self._width - self.map_buffer,
            )
            ranges[vexpr.Variable(f"pointmess{i}_y")] = (
                0 + self.map_buffer,
                self._height - self.map_buffer,
            )
        return CompactSet(ranges)

    def _hazards(self, state) -> np.ndarray:
        """
        Returns *all* of the hazard locations, even those not on the board.
        The returned array is a view of state, so hazards can be modified in-place.
        """
        offset = self._obs_start_idx
        hazard_locations = state[offset : offset + self.num_obstacles * 2]
        assert len(hazard_locations) % 2 == 0
        return hazard_locations

    def _pointmesses(self, state) -> np.ndarray:
        """
        Returns *all* of the pointmess locations, even those not on the board.
        The returned array is a view of state, so pointmesses can be modified in-place.
        """
        offset = self._obs_start_idx + self.num_obstacles * 2
        pointmess_locations = state[offset:]
        assert len(pointmess_locations) % 2 == 0
        return pointmess_locations

    def _placed_pointmesses(self):
        """returns the positions of all placed pointmesses as a flat array containing the x coordinates at even indices and the y coordinates at odd indices. """
        pointmesses = self._pointmesses(self._state)
        return pointmesses[~np.isnan(pointmesses)]

    def _num_placed_pointmesses(self):
        return len(self._placed_pointmesses())

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        :param action: w, a
        """
        if self._done:
            EpisodeTerminatedException()
        assert action in self.action_space
        w, a = action

        v0 = self._state[self._v_idx]
        new_v = v0 + a * self.T
        # enforce the bounds on v ([0, 1]) by changing a.
        if new_v < 0:
            a = -v0 / self.T
        elif new_v > self.max_v:
            a = (self.max_v - v0) / self.T

        theta0 = self._state[self._theta_idx]
        new_state = self._state.copy()

        tw = self.T * w
        theta = theta0 + tw
        # keep theta in [-2 * pi, 2 * pi]
        if theta > 2 * pi:
            theta -= 2 * pi
        elif theta < -2 * pi:
            theta += 2 * pi

        cos_theta0 = cos(theta0)
        sin_theta0 = sin(theta0)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        new_state[self._w_idx] = w
        new_state[self._v_idx] += a * self.T
        new_state[self._theta_idx] = theta

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

        self._step += 1

        # compute the new reward before adding or removing any pointmesses.
        reward, done_code = self._get_reward_and_done_code(
            self._state, action, new_state
        )

        # add any new pointmesses caused by collisions.
        self._place_new_pointmesses(new_state)

        # remove any pointmesses the ego is touching.
        self._remove_colliding_pointmesses(new_state)

        # reaching a hazard doesn't mean you're done, but we record it in the done code
        # to use it in checking safety
        self._done = done_code not in (NOT_DONE, REACHED_HAZARD)
        obs = self._get_obs(np.array([new_v, cos_theta, sin_theta], dtype=np.float32))
        self._state = new_state

        return (
            obs,
            reward,
            self._done,
            {"done_reason": done_code, "unsafe": done_code == REACHED_HAZARD},
        )

    def _get_reward_and_done_code(self, state1, action, state2) -> Tuple[float, int]:
        """
        :param state:
        :return: (reward, done_code). Use _final_state_description to interpret done_code.
        """
        egox, egoy, goalx, goaly = state2[:4]

        if self._collision(state2):
            # TODO: change this to check for the goal if num_pointmesses_on_collision
            # could be 0
            return -1, REACHED_HAZARD

        colliding_pointmesses = self._at_pointmess(state2)
        if colliding_pointmesses:
            # TODO: should technically check whether you've just picked up the last
            # pointmesses and are at the goal. Or we can just say you have to be at
            # the goal in the step after all the pointmesses are gone
            return len(colliding_pointmesses), NOT_DONE

        if not (
            self.map_buffer <= egox <= self._width - self.map_buffer
            and self.map_buffer < egoy < self._height - self.map_buffer
        ):
            return -1, MOVED_OFF_MAP

        at_goal = self._circle_contains(egox, egoy, goalx, goaly, self._hazard_radius)
        num_pointmesses = self._num_placed_pointmesses()
        if at_goal and num_pointmesses == 0:
            return 1, REACHED_GOAL
        if self._step >= self.horizon:
            return 0, REACHED_STEP_LIMIT
        return 0, NOT_DONE

    def _remove_colliding_pointmesses(self, state):
        egox, egoy = state[:2]
        pointmesses = self._pointmesses(state).reshape(-1, 2)
        # the vectorized version doesn't play well with the nan locations
        # ego_xy = state[:2]
        # collision_idx = ((pointmesses - ego_xy) ** 2).sum(-1) <= self._pointmess_radius ** 2
        # pointmesses[collision_idx] = float("nan")
        for i, (px, py) in enumerate(pointmesses):
            if isnan(px):
                continue
            if self._circle_contains(egox, egoy, px, py, self._pointmess_radius):
                pointmesses[i] = float("nan")

    @property
    def _pointmess_idx_offset(self):
        return self._obs_start_idx + self.num_obstacles * 2

    def _is_valid_state(self, state):
        return state in self.oracle_space

    def _at_pointmess(self, state):
        """
        :param state: the state to example.
        :return: The x,y coordinates of the pointmness(es) that the robot is currently at; might be an empty list.
        """
        at = []
        egox, egoy = state[:2]
        pointmesses = self._placed_pointmesses()
        for i in range(0, len(pointmesses), 2):
            px, py = pointmesses[i], pointmesses[i + 1]
            if self._circle_contains(egox, egoy, px, py, self._pointmess_radius):
                at.append(px)
                at.append(py)
        return at

    def _final_state_description(self, done):
        if done == MOVED_OFF_MAP:
            return "moved off map"
        elif done == REACHED_GOAL_HAZARD:
            return "reached goal and hazard at the same time"
        elif done == REACHED_GOAL:
            return "reached goal and there are no pointmesses."
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

    def _collision(self, state):
        return self._find_collision(state) is not None

    def _find_collision(self, state) -> Optional[Tuple[int, int]]:
        egox, egoy = state[:2]
        for i in range(self.num_obstacles):
            xidx = self._obs_start_idx + 2 * i
            yidx = xidx + 1
            ox, oy = state[xidx], state[yidx]
            if not isnan(ox) and self._circle_contains(
                egox, egoy, ox, oy, self._hazard_radius
            ):
                return ox, oy
        return None

    def _all_collisions(self, state) -> List[Tuple[int, float, float]]:
        """ returns a 3-tuple: (haznum, obstacle x, obstacle y) where haznum is the obstacle number. """
        rv = []
        egox, egoy = state[:2]
        for i in range(self.num_obstacles):
            xidx = self._obs_start_idx + 2 * i
            yidx = xidx + 1
            ox, oy = state[xidx], state[yidx]
            if self._circle_contains(egox, egoy, ox, oy, self._hazard_radius):
                rv.append([i, ox, oy])
        return rv

    def _place_new_pointmesses(self, state):
        """Places new pointmess in state in-place."""
        for haznum, _, _ in self._all_collisions(state):
            hazard_xidx = self._obs_start_idx + 2 * haznum
            hazard_yidx = hazard_xidx + 1
            # the starting index of the pointmesses associated with this pointmess
            hazards_pointmesses_idx = (
                self._obs_start_idx
                + self.num_obstacles * 2
                + self.num_initial_pointmess * 2
                + haznum * 2
            )
            for i, pxpy in enumerate(
                self._compute_pointmess_placements(
                    state[hazard_xidx], state[hazard_yidx]
                )
            ):
                px, py = pxpy
                pmxidx = hazards_pointmesses_idx + 2 * i
                pmyidx = pmxidx + 1
                state[pmxidx] = px
                state[pmyidx] = py
            state[hazard_xidx] = float("NaN")
            state[hazard_yidx] = float("NaN")

    def _compute_pointmess_placements(self, x, y) -> List[Tuple[(float, float)]]:
        # divide the circle around the hazard into N equivalently sized regions so that the pointmesses would be
        # evenly distributed. Assert that we have enough space to place self.num_pointmesses_on_collision.
        # finally
        hazard_x_size, hazard_y_size = self._objs["hazard"][0].img.size
        # todo this is the place where we define the locations of new pointmesses.
        # needs some work.
        placement_options = [
            (x - hazard_x_size - 20, y),
            (x + hazard_x_size + 20, y),
            (x, y - hazard_y_size - 20),
            (x, y + hazard_y_size + 20),
        ]
        assert len(placement_options) >= self.num_pointmesses_on_collision
        random.shuffle(placement_options)
        choices = []
        for _ in range(self.num_pointmesses_on_collision):
            choices.append(placement_options.pop())
        return choices

    def _circle_contains(self, x, y, c_x, c_y, c_radius):
        return (x - c_x) ** 2 + (y - c_y) ** 2 <= c_radius ** 2

    def reset(self) -> np.ndarray:
        # start with NaNs so the unplaced pointmesses are all nan
        state = np.full_like(self._state, float("nan"))

        # start with no rotational or translational velocity.
        state[self._w_idx] = 0
        state[self._v_idx] = 0
        angle = random.random() * 2 * pi
        state[self._theta_idx] = angle

        # place objects so they don't collide
        points = gen_separated_points(
            self.num_obstacles + self.num_initial_pointmess + 2,
            sep=self._hazard_radius,
            lower_bounds=np.array([self.map_buffer, self.map_buffer]),
            upper_bounds=np.array(
                [self._width - self.map_buffer, self._height - self.map_buffer]
            ),
        )
        state[self._ego_x_idx] = points[0, 0]
        state[self._ego_y_idx] = points[0, 1]
        state[self._goal_x_idx] = points[1, 0]
        state[self._goal_y_idx] = points[1, 1]
        points = points[2:].ravel()
        state[self._obs_start_idx : self._obs_start_idx + len(points)] = points

        # can't do this check because unplaced pointmesses are (nan, nan)
        # assert self._is_valid_state(state), self.oracle_space.to_state(state)
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
            vexpr.Variable("num_obstacles"): vexpr.Number(self.num_obstacles),
        }

    @staticmethod
    def constraint_func(action, sym_feats, num_obstacles, B, T, safe_sep):
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
        v = sym_feats[Pointmesses._v_idx]

        safe_dist = (
            (v ** 2 / (2 * B)) + ((a / B + 1) * (a / 2 * T ** 2 + T * v))
        ) + safe_sep

        start_idx = Pointmesses._obs_start_idx
        for i in range(start_idx, start_idx + 2 * num_obstacles, 2):
            ox = sym_feats[i]
            if isnan(ox):  # once one object is nan, all others will be
                break
            oy = sym_feats[i + 1]
            if abs(x - ox) < safe_dist and abs(y - oy) < safe_dist:
                return False
        return True

    @staticmethod
    def constrained_sample(sym_feats, num_obstacles, min_w, max_w, B, A, T, safe_sep):
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
        v = sym_feats[Pointmesses._v_idx]

        a = T ** 2 / (2 * B)
        b = T * v / B + T ** 2 / 2
        c0 = T * v + v ** 2 / (2 * B) + safe_sep

        safe_sets = portion.closed(-B, A)
        start_idx = Pointmesses._obs_start_idx
        for i in range(start_idx, start_idx + 2 * num_obstacles, 2):
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
