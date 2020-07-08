#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
There are only minor changes required in the Collector classes; we just have to update
all calls to `env.step` to pass in the sym_features from agent_info. The lines that
were changed have `# @changed` above them so that you can easily update these classes
if there are changes in `rlpyt`.
"""

import numpy as np
from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuEvalCollector, GpuResetCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.buffer import buffer_from_example, numpify_buffer, torchify_buffer


class SafeCpuResetCollector(CpuResetCollector):
    """
    Overrides `collect_batch` to pass the sym features from agent_info to the env.
    """

    safe: bool = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action, agent_info_np = numpify_buffer((act_pyt, agent_info))
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                # @changed
                if self.safe:
                    o, r, d, env_info = env.step(
                        action[b], agent_info_np.sym_features[b]
                    )
                else:
                    o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(
                    observation[b], action[b], r, d, agent_info[b], env_info
                )
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t].flat = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos


class SafeSerialEvalCollector(SerialEvalCollector):

    safe: bool = True

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(
            self.envs[0].action_space.null_value(), len(self.envs)
        )
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                # @changed
                if self.safe:
                    o, r, d, env_info = env.step(action[b], agent_info.sym_features[b])
                else:
                    o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(
                    observation[b], action[b], r, d, agent_info[b], env_info
                )
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (
                self.max_trajectories is not None
                and len(completed_traj_infos) >= self.max_trajectories
            ):
                # logger.log("Evaluation reached max num trajectories "
                # f"({self.max_trajectories}).")
                break
        # if t == self.max_T - 1:
        #     logger.log("Evaluation reached max num time steps "
        #         f"({self.max_T}).")
        return completed_traj_infos


class SafeGpuResetCollector(GpuResetCollector):

    safe: bool = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """Params agent_inputs and itr unused."""
        act_ready, obs_ready = self.sync.act_ready, self.sync.obs_ready
        step = self.step_buffer_np
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        env_buf.prev_reward[0] = step.reward
        obs_ready.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_ready.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                # @changed
                if self.safe:
                    o, r, d, env_info = env.step(
                        step.action[b], step.agent_info.sym_features[b]
                    )
                else:
                    o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(
                    step.observation[b],
                    step.action[b],
                    r,
                    d,
                    step.agent_info[b],
                    env_info,
                )
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = step.action  # OPTIONAL BY SERVER
            env_buf.reward[t] = step.reward
            env_buf.done[t] = step.done
            if step.agent_info:
                agent_buf.agent_info[t] = step.agent_info  # OPTIONAL BY SERVER
            obs_ready.release()  # Ready for server to use/write step buffer.

        return None, traj_infos, completed_infos


class SafeGpuEvalCollector(GpuEvalCollector):

    safe: bool = True

    def collect_evaluation(self, itr):
        """Param itr unused."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        act_ready, obs_ready = self.sync.act_ready, self.sync.obs_ready
        step = self.step_buffer_np
        for b, env in enumerate(self.envs):
            step.observation[b] = env.reset()
        step.done[:] = False
        obs_ready.release()

        for t in range(self.max_T):
            act_ready.acquire()
            if self.sync.stop_eval.value:
                obs_ready.release()  # Always release at end of loop.
                break
            for b, env in enumerate(self.envs):
                # @changed
                if self.safe:
                    o, r, d, env_info = env.step(
                        step.action[b], step.agent_info.sym_features[b]
                    )
                else:
                    o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(
                    step.observation[b],
                    step.action[b],
                    r,
                    d,
                    step.agent_info[b],
                    env_info,
                )
                if getattr(env_info, "traj_done", d):
                    self.traj_infos_queue.put(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
            obs_ready.release()
        self.traj_infos_queue.put(None)  # End sentinel.
