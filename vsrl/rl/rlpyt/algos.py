#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from collections import namedtuple

import torch
from rlpyt.algos.pg.ppo import PPO as PPO_
from rlpyt.algos.qpg.sac import SAC as SAC_
from rlpyt.utils.buffer import buffer_method
from rlpyt.utils.tensor import valid_mean

OptInfo = namedtuple(
    "OptInfo",
    [
        "q1Loss",
        "q2Loss",
        "piLoss",
        "q1GradNorm",
        "q2GradNorm",
        "piGradNorm",
        "q1",
        "q2",
        "piMu",
        "piLogStd",
        "qMeanDiff",
        "alpha",
    ],
)


class PPO(PPO_):
    def loss(
        self,
        agent_inputs,
        action,
        return_,
        advantage,
        valid,
        old_dist_info,
        init_rnn_state=None,
    ):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(
            action, old_dist_info=old_dist_info, new_dist_info=dist_info
        )
        ratio = ratio.clamp_max(1000)  # added (to prevent ratio == inf)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity


class SAC(SAC_):
    def initialize(
        self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0
    ):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = int(
            self.replay_ratio * sampler_bs / self.batch_size
        )
        print(
            f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration."
        )
        self.min_itr_learn = self.min_steps_learn // sampler_bs
        agent.give_min_itr_learn(self.min_itr_learn)
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            losses, values = self.loss(samples_from_replay)
            q1_loss, q2_loss, pi_loss, alpha_loss = losses

            if alpha_loss is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._alpha = torch.exp(self._log_alpha.detach())

            self.pi_optimizer.zero_grad()
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()

            combined_loss = q1_loss + q2_loss + pi_loss
            combined_loss.backward()

            # pi_loss.backward()
            # q1_loss.backward()
            # q2_loss.backward()

            pi_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.pi_parameters(), self.clip_grad_norm
            )
            self.pi_optimizer.step()

            # Step Q's last because pi_loss.backward() uses them?
            q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.q1_parameters(), self.clip_grad_norm
            )
            self.q1_optimizer.step()

            q2_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.q2_parameters(), self.clip_grad_norm
            )
            self.q2_optimizer.step()

            grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm)

            self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

        return opt_info
