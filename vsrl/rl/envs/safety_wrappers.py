import pickle
import random
from functools import partial
from pathlib import Path
from typing import Optional

import gym
import numpy as np


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


def wrap_environment(GymEnv):
    """
    Create a wrapper class for GymEnv which enforces safety.
    :param GymEnv: must have a `constraint_func`
    :returns: `SafeGymEnv` - a wrapper which uses the constraint to only allow
      `GymEnv.step` to be called with safe actions.
    """

    if not hasattr(GymEnv, "constraint_func"):
        raise ValueError("To be safe, `GymEnv` must have a `constraint_func`.")

    class SafeGymEnv(GymEnv):
        def __init__(
            self,
            *args,
            fallback_action: Optional[np.ndarray] = None,
            oracle_safety: bool = False,
            log_unsafe_transitions: bool = False,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if isinstance(self.action_space, gym.spaces.Discrete):
                self.constrained_sample = partial(
                    constrained_sample_finite,
                    constraint=self.constraint_func,
                    actions=[np.array([i]) for i in range(self.action_space.n)],
                )
            elif isinstance(self.action_space, gym.spaces.Box):
                if hasattr(self, "constrained_sample"):
                    self.constrained_sample = self.constrained_sample
                else:
                    self.constrained_sample = partial(
                        constrained_sample_continuous,
                        constraint=self.constraint_func,
                        lower_bounds=self.action_space.low,
                        upper_bounds=self.action_space.high,
                    )
            else:
                raise ValueError(
                    f"Do not know how to handle action space {self.action_space}."
                )

            self._log_unsafe_transitions = log_unsafe_transitions
            self._fallback_action = fallback_action

        def step(self, action, sym_features=None):
            constraint_used = False
            if sym_features is not None:
                sym_features = sym_features.squeeze()
                if not self.constraint_func(action, sym_features):
                    constraint_used = True
                    action = self.constrained_sample(sym_features)
                    if action is None:
                        if self._fallback_action is None:
                            raise ValueError(
                                "No safe action found! Consider adding fallback."
                            )
                        action = self._fallback_action

                # prepare this information in case the action isn't safe
                if self._log_unsafe_transitions:
                    unsafe_info = {
                        "sym_feats": sym_features.copy(),
                        # "img": self.render(),
                        "action": action.copy(),
                        "constraint_used": constraint_used,
                    }

            obs, reward, done, info = super().step(action)
            if (
                self._log_unsafe_transitions
                and info["action_unsafe"]
                and (sym_features is not None)
            ):
                debug_dir = Path.home() / "debug"
                debug_dir.mkdir(exist_ok=True)
                i = str(np.random.randint(1000))
                (debug_dir / f"{i}.pkl").write_bytes(pickle.dumps(unsafe_info))
            return obs, reward, done, info

    return SafeGymEnv


def wrap_symbolic_observation_env(Env):
    """
    The VSRL framework's safety wrapper for environments expects that each `env.step`
    call will provide the current symbolic (high-level) features in addition to the
    action. If an environment already returns the symbolic features as its observations,
    this wrapper can be used to pass those in automatically.
    """

    class SymbolicObsEnvWrapper(Env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._sym_features = None

        def reset(self):
            obs = super().reset()
            self._sym_features = obs
            return obs

        def step(self, action):
            obs, reward, done, info = super().step(action, self._sym_features)
            self._sym_features = obs
            return obs, reward, done, info

    return SymbolicObsEnvWrapper
