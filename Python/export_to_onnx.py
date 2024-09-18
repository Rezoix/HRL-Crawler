import argparse
import torch
from ray.rllib.policy.policy import Policy
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    default="C:\\Users\\Saku\\ray_results\\PPO_2023-11-20_13-03-28\\PPO_unity3d_6f147_00000_0_2023-11-20_13-03-28\\checkpoint_000053",
)


def convert(policy, filename):
    model = policy.model
    # input_shapes = [s.shape for s in model.config.observation_space.original_space.spaces]
    # input_shapes = (1,) + input_shapes
    # obs = torch.zeros(input_shapes)
    # dummy_inputs = {"obs": obs}
    # dummy_inputs = np.array([torch.zeros(s) for s in input_shapes])

    input_shape = model.config.observation_space.shape
    input_shape = (1,) + input_shape
    obs = torch.zeros(input_shape)
    dummy_inputs = {"obs": obs}

    torch.onnx.export(
        policy.model,
        (dummy_inputs),
        filename,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["output"],
        dynamic_axes={"obs": {0: "batch_size"}},
    )


def export_vanilla(policy, file_name, onnx_version):
    state_ins = [torch.zeros(1)]
    input_shape = policy.model.config.observation_space.shape
    input_shape = (1,) + input_shape  # Add dimension for batching
    obs = torch.zeros(input_shape)
    dummy_inputs = {"obs": obs}
    torch.onnx.export(
        policy.model,
        (dummy_inputs),
        file_name,
        export_params=True,
        opset_version=onnx_version,
        do_constant_folding=True,
        input_names=["obs", "unused_in"],
        output_names=["output", "unused_out"],
        dynamic_axes={"obs": {0: "batch_size"}},
    )


if __name__ == "__main__":
    args = parser.parse_args()

    chkp = Policy.from_checkpoint(args.path)
    policy = chkp["Crawler"]
    export_vanilla(policy, "Crawler.onnx", 12)
