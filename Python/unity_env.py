from gymnasium.spaces import Box, MultiDiscrete, Space, Tuple as TupleSpace
import gymnasium as gym
import logging
import numpy as np
import random
import time
from typing import Callable, Optional, Tuple, Dict, Any, List

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.utils.annotations import PublicAPI
# from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID

logger = logging.getLogger(__name__)

# RLlib typings
AgentID = str
MultiAgentDict = Dict[str, Any]


class MultiAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()

        if not hasattr(self, "observation_space"):
            self.observation_space = None
        if not hasattr(self, "action_space"):
            self.action_space = None
        if not hasattr(self, "_agent_ids"):
            self._agent_ids = set()


# @PublicAPI
class BetterUnity3DEnv(MultiAgentEnv):
    """A MultiAgentEnv representing a single Unity3D game instance.
    For an example on how to use this Env with a running Unity3D editor
    or with a compiled game, see:
    `rllib/examples/unity3d_env_local.py`
    For an example on how to use it inside a Unity game client, which
    connects to an RLlib Policy server, see:
    `rllib/examples/serving/unity3d_[client|server].py`
    Supports all Unity3D (MLAgents) examples, multi- or single-agent and
    gets converted automatically into an ExternalMultiAgentEnv, when used        self._previous_decision_step = None
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
        soft_horizon: bool = True,
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
            port_ = port or (
                self._BASE_PORT_ENVIRONMENT if file_name else self._BASE_PORT_EDITOR
            )
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
                    no_graphics=no_graphics,
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
            self.observation_space = Box(
                -high, high, shape=(self.n_agents, self.obs_dim), dtype=np.float32
            )

    def _get_vec_obs_size(self):
        res = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                res += obs_spec.shape[0]
        return res

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
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
            if self.api_version[0] > 1 or (
                self.api_version[0] == 1 and self.api_version[1] >= 4
            ):
                actions = action_dict
                """ for agent_id in self.unity_env.get_steps(behavior_name)[0].agent_id:
                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    # print(key)
                    # print(action_dict)
                    if key not in action_dict:
                        print("nokey")
                    else:
                        actions.append(action_dict[key]) """
                if actions.size > 0:
                    if actions[0].dtype == np.float32:
                        action_tuple = ActionTuple(continuous=np.array(actions))
                    else:
                        action_tuple = ActionTuple(discrete=np.array(actions))
                    self.unity_env.set_actions(behavior_name, action_tuple)
            # Old behavior: Do not use an ActionTuple and set each agent's
            # action individually.
            else:
                for agent_id in self.unity_env.get_steps(behavior_name)[
                    0
                ].agent_id_to_index.keys():
                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    self.unity_env.set_action_for_agent(
                        behavior_name, agent_id, action_dict[key]
                    )
        # Do the step.
        self.unity_env.step()

        obs, rewards, terminateds, truncateds, infos = self._get_step_results()

        # Global horizon reached? -> Return __all__ truncated=True, so user
        # can reset. Set all agents' individual `trunepisode_horizoncated` to True as well.
        self.episode_timesteps += 1
        if self.episode_timesteps >= self.episode_horizon:
            return (
                obs,
                rewards,
                terminateds,
                [
                    1 for _ in range(self.n_agents)
                ],  # dict({"__all__": 1}, **{agent_id: 1 for agent_id in all_agents}),
                infos,
            )

        return obs, rewards, terminateds, truncateds, infos

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        if not self.soft_horizon:
            self.unity_env.reset()
        obs, _, _, _, infos = self._get_step_results()
        return obs, infos

    def action_space_sample(self, agent_ids=None):
        if agent_ids == None:
            agent_ids = self.get_agent_ids()
        samples = np.empty((self.n_agents, self.action_size), dtype=np.float32)
        for agent_id in agent_ids:
            samples[agent_id] = self.action_space.sample()
        if len(samples) == 0:
            print("no agents -> no samples???")
        return samples

    def get_agent_ids(self):
        all_agents = []
        for behavior_name in self.unity_env.behavior_specs:
            for step in self.unity_env.get_steps(behavior_name):
                for agent_id in step.agent_id:
                    all_agents.append(agent_id)
        if len(all_agents) == 0:
            print("no agents???")
        return all_agents

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
        obs = np.empty((self.n_agents, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.n_agents, dtype=np.float32)
        terminateds = np.empty(self.n_agents, dtype=int)  # {"__all__": False}
        truncateds = np.empty(self.n_agents, dtype=int)
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
                # key = behavior_name + "_{}".format(agent_id)
                terminateds[agent_id] = 0

                # TMP
                truncateds[agent_id] = 0

                os = tuple(o[idx] for o in decision_steps.obs)
                os = (
                    os[0]
                    if len(os) == 1
                    else np.array(np.concatenate(os), dtype=np.float32)
                )  # Concatenate observations into single array
                obs[agent_id] = os
                rewards[agent_id] = (
                    decision_steps.reward[idx] + decision_steps.group_reward[idx]
                )
                # print(f"{key}, {rewards[key]}, {decision_steps.group_reward[idx]}")
                # print(i)
                i += 1
            for agent_id, idx in terminal_steps.agent_id_to_index.items():
                # key = behavior_name + "_{}".format(agent_id)
                terminateds[agent_id] = 1

                # TMP
                truncateds[agent_id] = 0

                # Only overwrite rewards (last reward in episode), b/c obs
                # here is the last obs (which doesn't matter anyways).
                # Unless key does not exist in obs.
                if agent_id not in obs:
                    os = tuple(o[idx] for o in terminal_steps.obs)
                    obs[agent_id] = os = (
                        os[0]
                        if len(os) == 1
                        else np.array(np.concatenate(os), dtype=np.float32)
                    )  # Concatenate observations into single array
                rewards[agent_id] = (
                    terminal_steps.reward[idx] + terminal_steps.group_reward[idx]
                )

        # Only use dones if all agents are done, then we should do a reset.
        # if False not in terminateds.values():
        #    terminateds["__all__"] = True
        # TODO: How to report that only one agent is done? RLlib seems to crash in this simple situation
        return obs, rewards, terminateds, truncateds, infos
        # return obs, rewards, terminateds, {"__all__": False}, infos
