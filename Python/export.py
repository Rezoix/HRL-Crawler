import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib import Policy
from ray.tune import Callback
from ray.tune.experiment import Trial

from unity_env import BetterUnity3DEnv

from torch import nn
import torch.onnx
import io
import numpy as np
import onnx


parser = argparse.ArgumentParser()

parser.add_argument(
    "--file-name",
    type=str,
    default="C:\\Users\\Saku\\ml-agents-release_20\\build\\UnityEnvironment.exe",
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)
parser.add_argument(
    "--path",
    type=str,
    default="C:\\Users\\Saku\\ray_results\\PPO\\PPO_unity3d_2f129_00000_0_2023-03-13_10-36-43\\checkpoint_000035",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)


class SaveCheckpointCallback(Callback):
    def __init__(self, agent_names, save_path) -> None:
        super().__init__()
        self.agent_names = agent_names
        self.save_path = save_path

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        super().on_checkpoint(iteration, trials, trial, checkpoint, **info)
        real_checkpoint = checkpoint.to_air_checkpoint()
        # mean_reward = checkpoint[1]['policy_reward_mean/TestAgentAudio']
        # # print(f"Checkpoint eval/mean: {mean_reward}")
        print(f"Exporting best models to {self.save_path}")
        for agent_name in self.agent_names:
            model = Policy.from_checkpoint(real_checkpoint)[agent_name]
            # fixed_model_path = save_path+f"/fixed_{mean_reward:.2f}.onnx"
            # # fixed_model_path = save_path+f"/fixed_{iteration}.onnx"
            fixed_model_path = self.save_path + f"/latest_{agent_name}.onnx"
            model.export_model(export_dir=self.save_path, onnx=12)
            convert(self.save_path + "/model.onnx", fixed_model_path)
            print(f"Exported fixed model to {fixed_model_path}")


def convert(model_path, fixed_model_path):
    torchmodel = onnx.load(model_path)  # the rllib output model dir
    onnx.checker.check_model(torchmodel)

    graph = torchmodel.graph

    for node in graph.node:
        if "Identity_14" in node.name or "Identity_13" in node.name:
            graph.node.remove(node)

    graph.input.pop()
    graph.input[0].name = "obs_0"
    # graph.node[1].input[0] = "obs_0"

    for node in graph.node:
        if "Cast_0" in node.name or "Cast_1" in node.name:
            node.input[0] = "obs_0"

    starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, [1], [0])
    ends = onnx.helper.make_tensor("ends", onnx.TensorProto.INT64, [1], [2])
    axes = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], [-1])
    graph.initializer.append(starts)
    graph.initializer.append(ends)
    graph.initializer.append(axes)

    version_number = onnx.helper.make_tensor("version_number", onnx.TensorProto.INT64, [1], [3])
    memory_size = onnx.helper.make_tensor("memory_size", onnx.TensorProto.INT64, [1], [0])
    continuous_action_output_shape = onnx.helper.make_tensor(
        "continuous_action_output_shape", onnx.TensorProto.INT64, [1], [2]
    )
    graph.initializer.append(version_number)
    graph.initializer.append(memory_size)
    graph.initializer.append(continuous_action_output_shape)

    node = onnx.helper.make_node(
        "Slice",
        inputs=["output", "starts", "ends", "axes"],
        outputs=["continuous_actions"],
    )
    graph.node.append(node)

    while len(graph.output):
        graph.output.pop()
    actions_info = onnx.helper.make_tensor_value_info("continuous_actions", onnx.TensorProto.FLOAT, shape=[])
    graph.output.append(actions_info)
    version_number_info = onnx.helper.make_tensor_value_info(
        "version_number", onnx.TensorProto.INT64, shape=[]
    )
    graph.output.append(version_number_info)
    memory_size_info = onnx.helper.make_tensor_value_info("memory_size", onnx.TensorProto.INT64, shape=[])
    graph.output.append(memory_size_info)
    continuous_action_output_shape_info = onnx.helper.make_tensor_value_info(
        "continuous_action_output_shape", onnx.TensorProto.INT64, shape=[]
    )
    graph.output.append(continuous_action_output_shape_info)

    onnx.checker.check_model(torchmodel)
    onnx.save(
        torchmodel,
        fixed_model_path,
    )


""" 
if __name__ == "__main__":
    ray.init()

    args = parser.parse_args()

    tune.register_env(
        "unity3d",
        lambda c: Env(
            file_name=c["file_name"],
            no_graphics=(c["file_name"] is not None),
            episode_horizon=c["episode_horizon"],
        ),
    )
    policies, policy_mapping_fn = Env.get_policy_configs_for_game()

    config = (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": args.file_name,
                "episode_horizon": args.horizon,
            },
            disable_env_checking=True,
        )
        .framework("torch")
        .training(
            model={"fcnet_hiddens": [512, 512, 512]},
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    trainer = PPOTrainer(config=config)
    trainer.restore(args.path)

    Policies = ["Player"]

    for pol in Policies:
        trainer.get_policy(pol).export_model("./models", onnx=12)
        convert("./models/model.onnx", "./models/fixed.onnx")
 """
