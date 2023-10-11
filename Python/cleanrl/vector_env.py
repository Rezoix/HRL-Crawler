from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from unity_env_vector import BetterUnity3DEnv as Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.sync_vector_env import SyncVectorEnv


class MultiSyncVectorEnv(SyncVectorEnv):
    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        super().__init__(
            env_fns=env_fns, observation_space=observation_space, action_space=action_space, copy=copy
        )

        self.n_agents = self.envs[0].n_agents

        self._rewards = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs, self.n_agents), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs, self.n_agents), dtype=np.bool_)

    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for env_id, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[env_id],
                self._terminateds[env_id],
                self._truncateds[env_id],
                info,
            ) = env.step(action)

            """ for agent_id in range(self.n_agents):
                if self._terminateds[env_id][agent_id] or self._truncateds[env_id][agent_id]:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info[agent_id]["final_observation"] = old_observation
                    info[agent_id]["final_info"] = old_info """
            observations.append(observation)
            # infos = self._add_info(infos, info, env_id)
        self.observations = concatenate(self.single_observation_space, observations, self.observations)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )
