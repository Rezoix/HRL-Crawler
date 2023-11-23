import argparse
import os

from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace

import numpy as np
import ray
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from unity_env import BetterUnity3DEnv, HRLUnityEnv
from hiro import HIROHigh, HIROLow

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="C:\\Users\\Saku\\ray_results\\PPO_2023-11-20_13-03-28\\PPO_unity3d_6f147_00000_0_2023-11-20_13-03-28\\checkpoint_000053",
)
parser.add_argument(
    "--file-name",
    type=str,
    default="..\\build\\Crawler.exe",
    help="The Unity3d binary (compiled) game filepath.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=1000,  # time_horizon in mlagents Crawler config
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    timescale = 1.0

    use_hrl = False

    if use_hrl:
        ModelCatalog.register_custom_model("HIROHigh", HIROHigh)
        ModelCatalog.register_custom_model("HIROLow", HIROLow)

        env = HRLUnityEnv(
            file_name=args.file_name,
            no_graphics=(args.file_name is not None),
            episode_horizon=args.horizon,
            timescale=timescale,
            high_level_steps=10,
        )

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("low_level_"):
                return "policy_low"
            else:
                return "policy_high"

        goal_vector_length = 10
        goal_vector_space = Box(0, 1, (goal_vector_length,))

        policies = {
            "policy_high": PolicySpec(
                observation_space=TupleSpace(
                    [
                        Box(-np.inf, np.inf, (126,)),
                        Box(-np.inf, np.inf, (51,)),
                    ]
                ),
                action_space=goal_vector_space,  # Goal vector
                config={
                    "model": {
                        "custom_model": "HIROHigh",
                        "custom_model_config": {"fc_size": 512},
                    }
                },
            ),
            "policy_low": PolicySpec(
                observation_space=TupleSpace(
                    [
                        Box(-np.inf, np.inf, (126,)),
                        Box(-np.inf, np.inf, (51,)),
                        goal_vector_space,  # Goal vector
                    ]
                ),
                action_space=Box(-1, 1, (20,)),
                config={
                    "model": {
                        "custom_model": "HIROLow",
                        "custom_model_config": {"fc_size": 512, "goal_size": goal_vector_length},
                    }
                },
            ),
        }

    else:  # Normal training without HRL
        env = BetterUnity3DEnv(
            file_name=args.file_name,
            no_graphics=(args.file_name is not None),
            episode_horizon=args.horizon,
            timescale=timescale,
        )

        # Get policies (different agent types; "behaviors" in MLAgents) and
        # the mappings from individual agents to Policies.
        policies, policy_mapping_fn = BetterUnity3DEnv.get_policy()

    algo = Policy.from_checkpoint(args.checkpoint)

    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset()

    while True:
        actions = {}
        for key, value in obs.items():
            ob = np.concatenate(value)
            act = algo["Crawler"].compute_single_action(ob)
            actions[key] = act[0]

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        episode_reward += sum(rewards.values())

        if terminateds["__all__"] or truncateds["__all__"]:
            print(f"EP reward: {episode_reward}")
            episode_reward = 0
            obs, info = env.reset()
