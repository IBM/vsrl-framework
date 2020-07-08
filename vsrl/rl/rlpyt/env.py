#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import inspect
import operator as op
import pickle
import random
from functools import partial
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from rlpyt.envs.base import Env as RLPytEnvBase
from rlpyt.envs.base import EnvStep
from rlpyt.samplers.collections import TrajInfo
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox

from vsrl.rl.envs.env import Env as VSRLEnv
from vsrl.spaces import CompactSet, FiniteSpace
from vsrl.verifier import expr, traversal
from vsrl.verifier.expr_helpers import multi_replace

NUMBER_OPS = {"Plus": op.add, "Times": op.mul, "Minus": op.sub, "Divide": op.truediv}
BOOL_OPS = {
    "Greater": op.gt,
    "GreaterEq": op.ge,
    "Less": op.lt,
    "LessEq": op.le,
    "And": op.and_,
    "Or": op.or_,
}


class SafetyEnvInfo(NamedTuple):
    """Info collected at each `step` of the env."""

    action_unsafe: bool  # whether an unsafe action was actually taken
    # the amount of penalty applied if an unsafe action was taken (otherwise, 0)
    unsafe_penalty: float
    # whether a constraint was used (which should prevent unsafe actions)
    constraint_used: bool


class SafetyEnvTrajInfo(TrajInfo):
    """
    Stores information about one trajectory (episode) of RL for logging.

    Any attributes not prefixed by an underscore will be logged.
    """

    def __init__(self):
        super().__init__()
        self.n_unsafe_actions = 0
        self.reward_ignoring_safety = 0
        self.constraint_used = 0

    def step(
        self,
        observation: np.ndarray,
        action,
        reward: float,
        done: bool,
        agent_info,
        env_info: SafetyEnvInfo,
    ) -> None:
        super().step(observation, action, reward, done, agent_info, env_info)
        self.reward_ignoring_safety += reward - env_info.unsafe_penalty
        self.n_unsafe_actions += env_info.action_unsafe
        self.constraint_used += env_info.constraint_used


def constrained_sample_finite(current_state, constraint, actions) -> np.ndarray:
    """Implements rejection sampling."""
    random.shuffle(actions)
    for action in actions:
        if constraint(action, current_state):
            return action
    return None


def constrained_sample_continuous(
    current_state, constraint, lower_bounds, upper_bounds
) -> np.ndarray:
    max_attempts = 10
    n_samples = 1000
    sample_size = (n_samples, len(lower_bounds))
    for _ in range(max_attempts):
        actions = np.random.uniform(lower_bounds, upper_bounds, size=sample_size)
        for action in actions:
            if constraint(action, current_state):
                return action
    return None


class RLPytEnv(RLPytEnvBase):
    """
    Continuous action spaces are converted to be [-1, 1]; conversion back to the original
    action space is done in `step` so that actions in the expected ranges are passed to
    the underlying environment.
    """

    def __init__(
        self,
        vsrl_env: VSRLEnv,
        oracle_safety: bool = False,
        log_unsafe_transitions: bool = False,
    ):
        """
        :param log_unsafe_transitions: only used when sym_features is given at `step`
        """
        super().__init__()
        self._env = vsrl_env
        action_space = vsrl_env.action_space

        if isinstance(action_space, FiniteSpace):
            # rlpyt discrete spaces can only be `IntBox`s, so I just make them
            # IntBox(0, n_actions) and then map the action back to our space in step
            n_actions = len(action_space.elements)
            self._action_space = IntBox(0, n_actions)
            # self._action_space = IntBox(0, n_actions, shape=(1,))
            self._actions = np.stack(action_space.elements)
            self.constrained_sample = partial(
                constrained_sample_finite,
                constraint=self._env.constraint_func,
                actions=action_space.elements[:],
            )
        elif isinstance(action_space, CompactSet):
            low, high = zip(*action_space.bounds.values())
            ones = np.ones_like(low)
            self._action_space = FloatBox(-ones, ones)
            self._actions = None
            self._action_lb = np.array(low, dtype=np.float32)
            self._action_ub = np.array(high, dtype=np.float32)
            self._action_range = self._action_ub - self._action_lb
            if hasattr(self._env, "constrained_sample"):
                self.constrained_sample = self._env.constrained_sample
            else:
                self.constrained_sample = partial(
                    constrained_sample_continuous,
                    constraint=self._env.constraint_func,
                    lower_bounds=self._action_lb,
                    upper_bounds=self._action_ub,
                )
        else:
            raise ValueError(
                f"Do not know how to convert action space {action_space} to rlpyt."
            )

        self._observation_space = vsrl_env.observation_space
        self.log_unsafe_transitions = log_unsafe_transitions
        self.oracle_safety = oracle_safety
        self._fallback_action = None

    def step(self, action, sym_features=None):
        """
        :param sym_features: if given, this is a safety-wrapped environment
        """
        constraint_used = False
        if self._actions is not None:
            action = self._actions[action]
        else:
            # continuous actions from agents are in [-1, 1]; convert back here
            action = (action + 1) / 2 * self._action_range + self._action_lb
        if sym_features is not None or self.oracle_safety:
            current_state = self._env.current_oracle_state()
            if sym_features is not None:
                sym_features = sym_features.squeeze()
                nan_idx = np.isnan(sym_features)
                sym_features[nan_idx] = current_state[nan_idx]
            else:  # oracle safety
                sym_features = current_state
            if not self._env.constraint_func(action, sym_features):
                constraint_used = True
                action = self.constrained_sample(sym_features)
                if action is None:
                    if self._fallback_action is None:
                        raise ValueError(
                            "No safe action found! Consider adding fallback."
                        )
                    action = self._fallback_action

            if self.log_unsafe_transitions:
                unsafe_info = {
                    "oracle": self._env.current_oracle_state(),
                    "sym_feats": sym_features.copy(),
                    "img": self._env.render(),
                    "action": action.copy(),
                    "constraint_used": constraint_used,
                }

        obs, reward, done, info = self._env.step(action)
        info = SafetyEnvInfo(info["unsafe"], 0, constraint_used)
        if (
            self.log_unsafe_transitions
            and info.action_unsafe
            and sym_features is not None
        ):
            debug_dir = Path.home() / "debug"
            debug_dir.mkdir(exist_ok=True)
            i = str(np.random.randint(1000))
            unsafe_info["oracle_next"] = self._env.current_oracle_state()
            (debug_dir / f"{i}.pkl").write_bytes(pickle.dumps(unsafe_info))
        return EnvStep(obs, reward, done, info)

    def reset(self):
        return self._env.reset()

    @property
    def horizon(self):
        return self._env.horizon()


def make_constraint_func(
    env: VSRLEnv, action_names: Sequence[str], sym_feat_names: Sequence[str]
):
    # e.g.
    # action_names = ["acc"]
    # sym_feat_names = ["follower_pos", "rvel", "leader_pos"]
    # self.constraint = make_constraint_func(vsrl_env, action_names, sym_feat_names)
    constraint = multi_replace(env.state_constants(), env.constraint)
    constraint = simplify(constraint)
    constraint = (
        str(constraint)
        .replace(" = ", " == ")
        .replace(" & ", " and ")
        .replace(" | ", " or ")
    )
    fmt = {"constraint": constraint}
    if len(action_names) == 1:
        fmt["action_name_str"] = action_names[0] + ","
    else:
        fmt["action_name_str"] = " ,".join(action_names)
    if len(sym_feat_names) == 1:
        fmt["sym_feat_name_str"] = sym_feat_names[0] + ","
    else:
        fmt["sym_feat_name_str"] = " ,".join(sym_feat_names)

    func_template = """
    def constraint(action, sym_features):
        {action_name_str} = action
        {sym_feat_name_str} = sym_features
        return {constraint}
    """
    func_str = func_template.format(**fmt)
    globals_ = {}
    exec(func_str.strip(), globals_)
    constraint = globals_["constraint"]
    constraint.__doc__ = fmt["constraint"]
    return constraint


def _simplify(node):
    if isinstance(node, expr.And):
        if isinstance(node.left, expr.Bool):
            if node.left:
                return node.right
            else:
                return expr.FalseF()
        if isinstance(node.right, expr.Bool):
            if node.right:
                return node.left
            else:
                return expr.FalseF()
    elif isinstance(node, expr.Or):
        if isinstance(node.left, expr.Bool):
            if node.left:
                return expr.TrueF()
            else:
                return node.right
        if isinstance(node.right, expr.Bool):
            if node.right:
                return expr.TrueF()
            else:
                return node.left
    else:
        try:
            if isinstance(node.left, expr.Number) and isinstance(
                node.right, expr.Number
            ):
                func = NUMBER_OPS.get(type(node).__name__)
                if func:
                    return expr.Number(func(node.left.val, node.right.val))
                func = BOOL_OPS.get(type(node).__name__)
                if func:
                    ret = func(node.left.val, node.right.val)
                    assert isinstance(ret, (bool, np.bool_))
                    return expr.TrueF() if ret else expr.FalseF()
            elif isinstance(node.left, expr.Bool) and isinstance(node.right, expr.Bool):
                func = BOOL_OPS.get(type(node).__name__)
                if func:
                    ret = func(bool(node.left), bool(node.right))
                    assert isinstance(ret, (bool, np.bool_))
                    return expr.TrueF() if ret else expr.FalseF()
        except AttributeError:
            pass
    return node


def simplify(expr):
    str_length = len(str(expr))
    while True:
        traversal.on_every_node(_simplify, expr)
        if len(str(expr)) == str_length:
            break
        str_length = len(str(expr))
    return expr
