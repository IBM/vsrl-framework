#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
The updates to the agents change the
* model inputs (to be just observations, not previous actions and rewards too)
* model outputs (to add sym_features)
* agent info (to add the sym_features so the Collector can pass these to the environment)
"""

import itertools

import numpy as np
import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.distributions.categorical import DistInfo
from rlpyt.distributions.gaussian import DistInfoStd, Gaussian
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from .models import ImpalaSacModel

MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
SafeAgentInfo = namedarraytuple("SafeAgentInfo", ["dist_info", "value", "sym_features"])
SafeSacAgentInfo = namedarraytuple("SafeAgentInfo", ["dist_info", "sym_features"])


class SafeCategoricalPgAgent(CategoricalPgAgent):
    """
    Overrides `step` to add symbolic features to `AgentStep.agent_info`.

    The `model` passed at init must return `pi, value, sym_features` where
    * `pi` is the policy distribution
    * `value` is the value function evaluation at the current state
    * `sym_features` are the symbolic features in the current state

    As inputs, it should just take observations (instead of observation, prev_action,
    prev_reward), though this could easily be reverted.

    `__call__` and `value` are  also overriden to work with the new model inputs / outputs.
    """

    def __call__(self, observation, prev_action=None, prev_reward=None):
        pi, value, _ = self.model(observation.to(device=self.device))
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action=None, prev_reward=None):
        pi, value, sym_features = self.model(
            observation.to(device=self.device), extract_sym_features=True
        )
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        # either sym_features should always be given or never
        if sym_features is not None:
            agent_info = SafeAgentInfo(
                dist_info=dist_info, value=value, sym_features=sym_features
            )
        else:
            agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action=None, prev_reward=None):
        model_inputs = buffer_to((observation,), device=self.device)[0]
        _, value, _ = self.model(model_inputs)
        return value.to("cpu")


class SafeGaussianPgAgent(GaussianPgAgent):
    """
    Overrides `step` to add symbolic features to `AgentStep.agent_info`.

    The `model` passed at init must return `mu, log_std, value, sym_features` where
    * `mu` and `log_std` parametrize the policy distribution
    * `value` is the value function evaluation at the current state
    * `sym_features` are the symbolic features in the current state

    As inputs, it should just take observations (instead of observation, prev_action,
    prev_reward), though this could easily be reverted.

    `__call__` and `value` are  also overriden to work with the new model inputs / outputs.
    """

    def __call__(self, observation, prev_action, prev_reward):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation,), device=self.device)[0]
        mu, log_std, value, _ = self.model(model_inputs)
        return buffer_to((DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation,), device=self.device)[0]
        mu, log_std, value, sym_features = self.model(
            model_inputs, extract_sym_features=True
        )
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        action = self.distribution.sample(dist_info)
        action = action.clamp(-1, 1)
        agent_info = SafeAgentInfo(
            dist_info=dist_info, value=value, sym_features=sym_features
        )
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state, e.g. for the
        bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
        """
        model_inputs = buffer_to((observation,), device=self.device)[0]
        _, _, value, _ = self.model(model_inputs)
        return value.to("cpu")

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        """Extends base method to build Gaussian distribution."""
        if (
            not (env_spaces.action.high == 1).all()
            and (env_spaces.action.low == -1).all()
        ):
            raise ValueError(f"The space for all actions should be [-1, 1].")
        super().initialize(
            env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks
        )
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0], min_std=1e-6, max_std=1
        )


class SafeSacAgent(SacAgent):
    """
    Agent for SAC algorithm, including action-squashing, using twin Q-values.

    Modifications:
    * prev_reward and prev_action aren't used

    Design decisions
    * The CNN parameters count as policy parameters; when updating Q1 / Q2, these are
      not updated.
    """

    def __init__(
        self,
        ModelCls=ImpalaSacModel,
        model_kwargs=None,
        initial_model_state_dict=None,
        action_squash=1.0,  # Max magnitude (or None).
        pretrain_std=0.75,  # With squash 0.75 is near uniform.
    ):
        """Saves input arguments; network defaults stored within."""
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[256, 256])
        super(SacAgent, self).__init__(
            ModelCls=ModelCls,
            model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict,
        )
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        super(SacAgent, self).initialize(
            env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks
        )

        self.target_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

    def to_device(self, cuda_idx=None):
        super(SacAgent, self).to_device(cuda_idx)
        self.target_model.to(self.device)

    def data_parallel(self):
        super(SacAgent, self).data_parallel()
        DDP_WRAP = DDPC if self.device.type == "cpu" else DDP
        self.target_model = DDP_WRAP(self.target_model)

    def q(self, observation, prev_action, prev_reward, action):
        """Compute twin Q-values for state/observation and input action (with grad)."""
        model_inputs = buffer_to((observation, action), device=self.device)
        q1, q2, _ = self.model(model_inputs, "q")
        return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward, action):
        """Compute twin target Q-values for state/observation and input action."""
        model_inputs = buffer_to((observation, action), device=self.device)
        target_q1, target_q2, _ = self.target_model(model_inputs, "q")
        return target_q1.cpu(), target_q2.cpu()

    def pi(self, observation, prev_action, prev_reward):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distribution, which handles action squashing
        through this process."""
        model_inputs = buffer_to(observation, device=self.device)
        mean, log_std, _ = self.model(model_inputs, "pi")
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to(observation, device=self.device)
        mean, log_std, sym_features = self.model(
            model_inputs, "pi", extract_sym_features=True
        )
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = SafeSacAgentInfo(dist_info=dist_info, sym_features=sym_features)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_model, self.model.state_dict(), tau)

    def pi_parameters(self):
        return itertools.chain(
            self.model.pi.parameters(), self.model.feature_extractor.parameters()
        )

    def q1_parameters(self):
        return self.model.q1.parameters()

    def q2_parameters(self):
        return self.model.q2.parameters()

    def train_mode(self, itr):
        super(SacAgent, self).train_mode(itr)
        self.target_model.train()

    def sample_mode(self, itr):
        super(SacAgent, self).sample_mode(itr)
        self.target_model.eval()
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super(SacAgent, self).eval_mode(itr)
        self.target_model.eval()
        self.distribution.set_std(0.0)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
