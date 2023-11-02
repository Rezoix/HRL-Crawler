from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace
import logging
import numpy as np
import random
import time
from typing import Callable, Optional, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID

logger = logging.getLogger(__name__)


@PublicAPI
class BetterUnity3DEnv(MultiAgentEnv):
    """A MultiAgentEnv representing a single Unity3D game instance.
    For an example on how to use this Env with a running Unity3D editor
    or with a compiled game, see:
    `rllib/examples/unity3d_env_local.py`
    For an example on how to use it inside a Unity game client, which
    connects to an RLlib Policy server, see:
    `rllib/examples/serving/unity3d_[client|server].py`
    Supports all Unity3D (MLAgents) examples, multi- or single-agent and
    gets converted automatically into an ExternalMultiAgentEnv, when used
    inside an RLlib PolicyClient for cloud/distributed training of Unity games.
    """

    # Default base port when connecting directly to the Editor
    _BASE_PORT_EDITOR = 5004
    # Default base port when connecting to a compiled environment
    _BASE_PORT_ENVIRONMENT = 5005
    # The worker_id for each environment instance
    _WORKER_ID = 0

    def __init__(
        self,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 30,
        episode_horizon: int = 1000,
        soft_horizon: bool = False,
        timescale: int = 1.0,
    ):
        """Initializes a Unity3DEnv object.
        Args:
            file_name (Optional[str]): Name of the Unity game binary.
                If None, will assume a locally running Unity3D editor
                to be used, instead.
            port (Optional[int]): Port number to connect to Unity environment.
            seed: A random seed value to use for the Unity3D game.
            no_graphics: Whether to run the Unity3D simulator in
                no-graphics mode. Default: False.
            timeout_wait: Time (in seconds) to wait for connection from
                the Unity3D instance.
            episode_horizon: A hard horizon to abide to. After at most
                this many steps (per-agent episode `step()` calls), the
                Unity3D game is reset and will start again (finishing the
                multi-agent episode that the game represents).
                Note: The game itself may contain its own episode length
                limits, which are always obeyed (on top of this value here).
        """

        super().__init__()

        if file_name is None:
            print(
                "No game binary provided, will use a running Unity editor "
                "instead.\nMake sure you are pressing the Play (|>) button in "
                "your editor to start."
            )

        import mlagents_envs
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )

        # Try connecting to the Unity3D game instance. If a port is blocked
        port_ = None
        while True:
            # Sleep for random time to allow for concurrent startup of many
            # environments (num_workers >> 1). Otherwise, would lead to port
            # conflicts sometimes.
            if port_ is not None:
                # time.sleep(random.randint(1, 10))
                time.sleep(random.random() * 0.5)
            port_ = port or (self._BASE_PORT_ENVIRONMENT if file_name else self._BASE_PORT_EDITOR)
            # cache the worker_id and
            # increase it for the next environment
            worker_id_ = BetterUnity3DEnv._WORKER_ID if file_name else 0
            BetterUnity3DEnv._WORKER_ID += 1
            seed = worker_id_  # TODO: Seed is hardcoded here, argument for parent function is ignored!
            print(f"Seed: {seed}")
            try:
                channel = EngineConfigurationChannel()
                self.unity_env = UnityEnvironment(
                    file_name=file_name,
                    worker_id=worker_id_,
                    base_port=port_,
                    seed=seed,
                    no_graphics=True if file_name is None else False,
                    timeout_wait=timeout_wait,
                    side_channels=[channel],
                )
                channel.set_configuration_parameters(time_scale=timescale)
                print("Created UnityEnvironment for port {}".format(port_ + worker_id_))
            except mlagents_envs.exception.UnityWorkerInUseException:
                pass
            else:
                break

        # ML-Agents API version.
        self.api_version = self.unity_env.API_VERSION.split(".")
        self.api_version = [int(s) for s in self.api_version]

        # Reset entire env every this number of step calls.
        self.episode_horizon = episode_horizon
        self.soft_horizon = soft_horizon
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0

        # Get env info
        if not self.unity_env.behavior_specs:
            self.unity_env.step()

        self.name = list(self.unity_env.behavior_specs.keys())[0]
        self.group_spec = self.unity_env.behavior_specs[self.name]

        # Check for num of agents
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.name)
        print(f"{len(decision_steps)} agents found in the environment")
        self.n_agents = len(decision_steps)
        self._previous_decision_step = decision_steps

        # Check action spaces (Only for continuous at the moment)
        if self.group_spec.action_spec.is_continuous():
            self.action_size = self.group_spec.action_spec.continuous_size
            high = 1
            self.action_space = Box(
                -high,
                high,
                shape=(self.n_agents, self.group_spec.action_spec.continuous_size),
                dtype=np.float32,
            )

        # Check observation spaces

        """ list_spaces: List[gym.Space] = []
        for obs_spec in self.group_spec.observation_specs:
            high = np.array([np.inf] * obs_spec.shape[0])
            list_spaces.append(Box(-high, high, dtype=np.float32))
        
        if len(list_spaces) > 1:
            self.observation_space = TupleSpace(list_spaces)
        else:
            self.observation_space = list_spaces[0] """

        self.obs_dim = self._get_vec_obs_size()
        if self.obs_dim > 0:
            high = np.inf
            self.observation_space = Box(-high, high, shape=(self.n_agents, self.obs_dim), dtype=np.float32)

        # self.observation_space = self._get_obs_shape()

    def _get_vec_obs_size(self):
        res = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                res += obs_spec.shape[0]
            elif len(obs_spec.shape) == 3:
                res += np.prod(obs_spec.shape)
        return res

    # Not currently used
    """ def _get_obs_shape(self):
        spaces = []
        for obs_spec in self.group_spec.observation_specs:
            # Vector observation
            if len(obs_spec.shape) == 1:
                spaces.append(Box(-1, 1, shape=obs_spec.shape, dtype=np.float32))
            # Grid observation
            if "GridSensor" in obs_spec.name:
                if "OneHot" in obs_spec.name:
                    spaces.append(Box(0, 1, shape=obs_spec.shape, dtype=int))
                if "Discrete" in obs_spec.name:
                    spaces.append(Box(0, obs_spec.shape[-1], shape=obs_spec.shape[0:2], dtype=int))
        ret = TupleSpace(spaces)
        shape = tuple([x.shape for x in spaces])
        ret._shape = shape
        return ret """

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Performs one multi-agent step through the game.
        Args:
            action_dict: Multi-agent action dict with:
                keys=agent identifier consisting of
                [MLagents behavior name, e.g. "Goalie?team=1"] + "_" +
                [Agent index, a unique MLAgent-assigned index per single agent]
        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                - infos: An (empty) info dict.
        """
        from mlagents_envs.base_env import ActionTuple

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        all_agents = []
        for behavior_name in self.unity_env.behavior_specs:
            # New ML-Agents API: Set all agents actions at the same time
            # via an ActionTuple. Since API v1.4.0.
            if self.api_version[0] > 1 or (self.api_version[0] == 1 and self.api_version[1] >= 4):
                actions = []
                for agent_id in self.unity_env.get_steps(behavior_name)[0].agent_id:
                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    # print(key)
                    # print(action_dict)
                    if key not in action_dict:
                        print("nokey")
                    else:
                        actions.append(action_dict[key])
                if actions:
                    if actions[0].dtype == np.float32:
                        action_tuple = ActionTuple(continuous=np.array(actions))
                    else:
                        action_tuple = ActionTuple(discrete=np.array(actions))
                    self.unity_env.set_actions(behavior_name, action_tuple)
            # Old behavior: Do not use an ActionTuple and set each agent's
            # action individually.
            else:
                for agent_id in self.unity_env.get_steps(behavior_name)[0].agent_id_to_index.keys():
                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    self.unity_env.set_action_for_agent(behavior_name, agent_id, action_dict[key])
        # Do the step.
        self.unity_env.step()

        obs, rewards, terminateds, truncateds, infos = self._get_step_results()

        # Global horizon reached? -> Return __all__ truncated=True, so user
        # can reset. Set all agents' individual `truncated` to True as well.
        self.episode_timesteps += 1
        if self.episode_timesteps >= self.episode_horizon:
            return (
                obs,
                rewards,
                terminateds,
                dict({"__all__": True}, **{agent_id: True for agent_id in all_agents}),
                infos,
            )

        return obs, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        if not self.soft_horizon:
            self.unity_env.reset()
        obs, _, _, _, infos = self._get_step_results()
        # print(obs)
        return obs, infos

    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.
        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        terminateds = {"__all__": False}
        infos = {}
        i = 0
        for behavior_name in self.unity_env.behavior_specs:
            decision_steps, terminal_steps = self.unity_env.get_steps(behavior_name)
            # Important: Only update those sub-envs that are currently
            # available within _env_state.
            # Loop through all envs ("agents") and fill in, whatever
            # information we have.
            # print(decision_steps.agent_id_to_index.items())
            for agent_id, idx in decision_steps.agent_id_to_index.items():
                key = behavior_name + "_{}".format(agent_id)
                terminateds[key] = False
                os = tuple(o[idx] for o in decision_steps.obs)
                os = os[0] if len(os) == 1 else os
                obs[key] = os
                rewards[key] = decision_steps.reward[idx] + decision_steps.group_reward[idx]
                # print(f"{key}, {rewards[key]}, {decision_steps.group_reward[idx]}")
                # print(i)
                i += 1
            for agent_id, idx in terminal_steps.agent_id_to_index.items():
                key = behavior_name + "_{}".format(agent_id)
                terminateds[key] = True
                # Only overwrite rewards (last reward in episode), b/c obs
                # here is the last obs (which doesn't matter anyways).
                # Unless key does not exist in obs.
                if key not in obs:
                    os = tuple(o[idx] for o in terminal_steps.obs)
                    obs[key] = os = os[0] if len(os) == 1 else os
                rewards[key] = terminal_steps.reward[idx] + terminal_steps.group_reward[idx]

        # Only use dones if all agents are done, then we should do a reset.
        if False not in terminateds.values():
            terminateds["__all__"] = True
            # TODO: How to report that only one agent is done? RLlib seems to crash in this simple situation
        return obs, rewards, {"__all__": False}, {"__all__": False}, infos
        # return obs, rewards, terminateds, {"__all__": False}, infos

    def get_policy() -> Tuple[dict, Callable[[AgentID], PolicyID]]:
        # policy = PolicySpec(observation_space=self.observation_space, action_space=self.action_space)

        # Due to how RLlib works, it seems to be not possible to fetch spaces automatically?
        # Maybe create a temp environment, get the spaces and destroy that environment?
        policy = PolicySpec(
            observation_space=TupleSpace(
                [
                    Box(-np.inf, np.inf, (126,)),
                    Box(-np.inf, np.inf, (51,)),
                ]
            ),
            action_space=Box(-1, 1, (20,)),
        )

        def mapping_fn(agent_id, episode, worker, **kwargs):
            return "Crawler"

        return {"Crawler": policy}, mapping_fn


# https://github.com/ray-project/ray/blob/be2606e1cd41c9743e2b53222b8e029395984343/rllib/examples/env/windy_maze_env.py
# Above only works for single agent. Requires significant modification to make work in multi-agent environemnt
class HRLUnityEnv(BetterUnity3DEnv):
    def __init__(
        self,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 30,
        episode_horizon: int = 1000,
        soft_horizon: bool = False,
        timescale: int = 1.0,
        high_level_steps: int = 25,
    ):
        super().__init__(
            file_name,
            port,
            seed,
            no_graphics,
            timeout_wait,
            episode_horizon,
            soft_horizon,
            timescale,
        )
        self.max_high_level_steps = high_level_steps

        # Used for indexing dictionaries
        self._agent_ids = set(self.unity_env.get_steps(self.name)[0].agent_id)

    def reset(self, *, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self.episode_timesteps = 0

        self.current_goal = {id: np.zeros((10,), dtype=np.float32) for id in self._agent_ids}
        self.steps_remaining_at_level = {id: None for id in self._agent_ids}
        self.num_high_level_steps = {id: 0 for id in self._agent_ids}
        self.low_level_agent_id = {id: f"low_level_{0}" for id in self._agent_ids}
        self.cur_obs = {id: None for id in self._agent_ids}

        if not self.soft_horizon:
            self.unity_env.reset()
        _, _, _, _, infos = self._get_step_results()

        obs = {f"high_level_{id}": self.cur_obs[id] for id in self._agent_ids}

        # print("Reset Done")

        return obs, infos

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # print(f"Step start: {self.episode_timesteps}")
        # print(action_dict.keys())
        obs_h, rew_h, terminateds_h, truncateds_h, infos_h = self._high_level_step(action_dict)
        obs_l, rew_l, terminateds_l, truncateds_l, infos_l = self._low_level_step(action_dict)

        obs_h.update(obs_l)
        rew_h.update(rew_l)
        terminateds_h.update(terminateds_l)
        truncateds_h.update(truncateds_l)
        infos_h.update(infos_l)

        if False not in terminateds_h.values():
            terminateds_h["__all__"] = True

        if False not in truncateds_h.values():
            truncateds_h["__all__"] = True

        return obs_h, rew_h, terminateds_h, truncateds_h, infos_h

    def _high_level_step(self, action_dict):
        # The high level step, setting goal vector
        obs = {}
        rew = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        for key, action in action_dict.items():
            if "high_level" in key:
                agent_id = int(key.split("_")[-1])
                self.current_goal[agent_id] = action

                self.steps_remaining_at_level[agent_id] = self.max_high_level_steps
                self.num_high_level_steps[agent_id] += 1
                self.low_level_agent_id[
                    agent_id
                ] = f"low_level_{agent_id}_{self.num_high_level_steps[agent_id]}"
                # f"low_level_{agent_id}_{self.num_high_level_steps[agent_id]}"

                obs[self.low_level_agent_id[agent_id]] = self.cur_obs[agent_id] + (
                    self.current_goal[agent_id],
                )
                # [self.cur_obs[agent_id], self.current_goal[agent_id]]
                rew[self.low_level_agent_id[agent_id]] = 0
                terminateds[self.low_level_agent_id[agent_id]] = False
                truncateds[self.low_level_agent_id[agent_id]] = False

        return obs, rew, terminateds, truncateds, infos

    def _low_level_step(self, action_dict):
        from mlagents_envs.base_env import ActionTuple

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        all_agents = []
        for behavior_name in self.unity_env.behavior_specs:
            actions = []
            for agent_id in self.unity_env.get_steps(behavior_name)[0].agent_id:
                key = self.low_level_agent_id[agent_id]
                all_agents.append(key)

                if key not in action_dict:
                    actions.append(np.full((self.action_size), 0, dtype=np.float32))
                else:
                    self.steps_remaining_at_level[agent_id] -= 1
                    actions.append(action_dict[key])
            if actions:
                # force continuous actions for the moment
                if actions[0].dtype == np.float32:
                    action_tuple = ActionTuple(continuous=np.array(actions))
                else:
                    action_tuple = ActionTuple(discrete=np.array(actions))
                self.unity_env.set_actions(behavior_name, action_tuple)

        # Do the step.
        self.unity_env.step()

        obs, rewards, terminateds, truncateds, infos = self._get_step_results()

        # Global horizon reached? -> Return __all__ truncated=True, so user
        # can reset. Set all agents' individual `truncated` to True as well.
        self.episode_timesteps += 1
        if self.episode_timesteps >= self.episode_horizon:
            truncateds = dict(
                {"__all__": True}, **{self.low_level_agent_id[agent_id]: True for agent_id in self._agent_ids}
            )
            """ return (
                obs,
                rewards,
                terminateds,
                dict({"__all__": True}, **{agent_id: True for agent_id in all_agents}),
                infos,
            ) """

        return obs, rewards, terminateds, truncateds, infos

    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.
        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        terminateds = {"__all__": False}
        infos = {}
        i = 0
        for behavior_name in self.unity_env.behavior_specs:
            decision_steps, terminal_steps = self.unity_env.get_steps(behavior_name)
            # Important: Only update those sub-envs that are currently
            # available within _env_state.
            # Loop through all envs ("agents") and fill in, whatever
            # information we have.
            # print(decision_steps.agent_id_to_index.items())

            for agent_id, idx in decision_steps.agent_id_to_index.items():
                key = self.low_level_agent_id[agent_id]
                os = tuple(o[idx] for o in decision_steps.obs)
                os = os[0] if len(os) == 1 else os
                self.cur_obs[agent_id] = os
                obs[key] = os + (self.current_goal[agent_id],)

                rewards[key] = decision_steps.reward[idx] + decision_steps.group_reward[idx]
                i += 1

                if self.steps_remaining_at_level[agent_id] == 0:
                    # Terminate low level agent
                    terminateds[key] = True

                    key_high = f"high_level_{agent_id}"
                    obs[key_high] = self.cur_obs[agent_id]
                    rewards[key_high] = rewards[key]
                else:
                    terminateds[key] = False

            for agent_id, idx in terminal_steps.agent_id_to_index.items():
                key = self.low_level_agent_id[agent_id]
                terminateds[key] = True
                # Only overwrite rewards (last reward in episode), b/c obs
                # here is the last obs (which doesn't matter anyways).
                # Unless key does not exist in obs.
                if key not in obs:
                    os = tuple(o[idx] for o in terminal_steps.obs)
                    os = os[0] if len(os) == 1 else os
                    self.cur_obs[agent_id] = os
                    obs[key] = os + (self.current_goal[agent_id],)
                rewards[key] = terminal_steps.reward[idx] + terminal_steps.group_reward[idx]

                key_high = f"high_level_{agent_id}"
                obs[key_high] = self.cur_obs[agent_id]
                rewards[key_high] = rewards[key]

        # Only use dones if all agents are done, then we should do a reset.
        # if False not in terminateds.values():
        #    terminateds["__all__"] = True
        #    # TODO: How to report that only one agent is done? RLlib seems to crash in this simple situation

        return obs, rewards, terminateds, {"__all__": False}, infos
