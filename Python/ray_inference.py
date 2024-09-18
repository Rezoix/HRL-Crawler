import argparse
import os

from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace

import numpy as np
import ray
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy


from hiro import HIROHigh, HIROLow

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="C:\\Users\\Saku\\ray_results\\PPO_2024-05-27_00-19-06\\PPO_unity3d_95e23_00000_0_2024-05-27_00-19-07\\checkpoint_000199",
    # HRL Full obs "C:\\Users\\Saku\\ray_results\\PPO_2024-03-26_01-04-20\\PPO_unity3d_03b67_00000_0_2024-03-26_01-04-21\\checkpoint_000199",
    # HRL Split obs "C:\\Users\\Saku\\ray_results\\PPO_2024-04-29_00-13-12\\PPO_unity3d_1f658_00000_0_2024-04-29_00-13-13\\checkpoint_000199"
    # Baseline "C:\\Users\\Saku\\ray_results\\PPO_2024-03-16_13-56-46\\PPO_unity3d_43a80_00000_0_2024-03-16_13-56-46\\checkpoint_000199",
)
parser.add_argument(
    "--file-name",
    type=str,
    default="C:\\Users\\Saku\\Documents\\Dippa\\Crawler\\build\\Crawler.exe",
    help="The Unity3d binary (compiled) game filepath.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=10000,  # time_horizon in mlagents Crawler config
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    timescale = 20.0

    use_hrl = True
    use_split_obs = False

    if use_hrl:
        from unity_env_old import BetterUnity3DEnv, HRLUnityEnv

        ModelCatalog.register_custom_model("HIROHigh", HIROHigh)
        ModelCatalog.register_custom_model("HIROLow", HIROLow)

        env = HRLUnityEnv(
            file_name=args.file_name,
            no_graphics=(args.file_name is not None),
            episode_horizon=args.horizon,
            timescale=timescale,
            high_level_steps=10,
            split_obs=use_split_obs,
        )

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("low_level_"):
                return "policy_low"
            else:
                return "policy_high"

        goal_vector_length = 10
        goal_vector_space = Box(0, 1, (goal_vector_length,))

        if use_split_obs:
            policies = {
                "policy_high": PolicySpec(
                    observation_space=TupleSpace(
                        [
                            Box(-np.inf, np.inf, (33,)),
                        ]
                    ),
                    action_space=goal_vector_space,  # Goal vector
                ),
                "policy_low": PolicySpec(
                    observation_space=TupleSpace(
                        [
                            Box(-np.inf, np.inf, (126,)),
                            Box(-np.inf, np.inf, (38,)),
                            goal_vector_space,  # Goal vector
                        ]
                    ),
                    action_space=Box(-1, 1, (20,)),
                ),
            }
        else:
            policies = {
                "policy_high": PolicySpec(
                    observation_space=TupleSpace(
                        [
                            Box(-np.inf, np.inf, (126,)),
                            Box(-np.inf, np.inf, (51,)),
                        ]
                    ),
                    action_space=goal_vector_space,  # Goal vector
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
                ),
            }

    else:  # Normal training without HRL
        from unity_env import BetterUnity3DEnv, HRLUnityEnv

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

    steps = 0
    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset()

    if use_hrl:
        while True:
            actions = {}
            for key, value in obs.items():
                if key.startswith("high_level_"):
                    ob = np.concatenate(value)
                    act = algo["policy_high"].compute_single_action(ob)
                    actions[key] = act[0]
                else:
                    ob = np.concatenate(value)
                    act = algo["policy_low"].compute_single_action(ob)
                    actions[key] = act[0]

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            steps += 1

            if terminateds["__all__"] or truncateds["__all__"] or steps > args.horizon:
                print(f"EP reward: {episode_reward}")
                episode_reward = 0
                steps = 0
                obs, info = env.reset()

    else:
        while True:
            actions = {}
            for key, value in obs.items():
                ob = np.concatenate(value)
                act = algo["Crawler"].compute_single_action(ob)
                actions[key] = act[0]

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            steps += 1

            if terminateds["__all__"] or truncateds["__all__"] or steps > args.horizon:
                print(f"EP reward: {episode_reward}")
                episode_reward = 0
                steps = 0
                obs, info = env.reset()
