import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

import torch
import torch.nn as nn


# https://github.com/tensorflow/models/blob/master/research/efficient-hrl/agent.py
# https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py


class HIRONetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.metaAgent = nn.Sequential()
        self.uvfAgent = nn.Sequential()

    @override(TorchModelV2)
    def forward(
        self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)

        metaOut = self.metaAgent(self._last_flat_in)
        uvfOut = self.uvfAgent(metaOut)

        logits = []
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return 0
