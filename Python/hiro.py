import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import override

import torch
import torch.nn as nn


# https://github.com/tensorflow/models/blob/master/research/efficient-hrl/agent.py
# https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py


class HIROHigh(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        fc_size: int = 512,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Flatten obs_space
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.hidden_width = 512

        self.fc = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
        )

        self.action_branch = nn.Sequential(
            nn.Linear(self.hidden_width, self.action_size),
            nn.Tanh(),
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.hidden_width, 1),
        )

        self._features = None

    @override(TorchModelV2)
    def forward(
        self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()  # .reshape(obs.shape[0], -1)
        self._features = self.fc(obs)
        goal_vector = self.action_branch(self._features)

        return goal_vector, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value = self.value_branch(self._features)
        return value


class HIROLow(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        fc_size: int = 512,
        goal_size: int = 5,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Flatten obs_space
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.hidden_width = 512

        self.fc = nn.Sequential(
            nn.Linear(self.obs_size + goal_size, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
        )

        self.action_branch = nn.Sequential(
            nn.Linear(self.hidden_width, self.action_size),
            nn.Tanh(),
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.hidden_width, 1),
        )

        self._features = None

    @override(TorchModelV2)
    def forward(
        self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()  # .reshape(obs.shape[0], -1)
        self._features = self.fc(obs)
        goal_vector = self.action_branch(self._features)

        return goal_vector, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value = self.value_branch(self._features)
        return value
