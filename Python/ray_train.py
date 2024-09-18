import argparse
import os

from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog

from ray.air.integrations.wandb import WandbLoggerCallback


from export import SaveCheckpointCallback
from hiro import HIROHigh, HIROLow

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=".\\models\\")
parser.add_argument(
    "--restore",
    type=str,
    default=None,  # "C:/Users/Saku/ray_results/PPO_2024-05-13_09-46-07",
    help="Filepath to checkpoint to restore training.",
)
parser.add_argument(
    "--file-name",
    type=str,
    default="C:\\Users\\Saku\\Documents\\Dippa\\Crawler\\build\\Crawler.exe",
    help="The Unity3d binary (compiled) game filepath.",
)

parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument("--stop-iters", type=int, default=9999, help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=500_000,  # 4_000_000,
    help="Number of timesteps to train. Measured in environment steps.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=99999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=1000,  # time_horizon in mlagents Crawler config
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)
parser.add_argument("--gpus", type=int, default=1, help="How many GPUs should be used.")

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_gpus=args.gpus)  # , local_mode=True)

    timescale = 20

    use_hrl = False
    use_split_obs = False

    if use_hrl:
        from unity_env_old import BetterUnity3DEnv, HRLUnityEnv

        ModelCatalog.register_custom_model("HIROHigh", HIROHigh)
        ModelCatalog.register_custom_model("HIROLow", HIROLow)

        tune.register_env(
            "unity3d",
            lambda c: HRLUnityEnv(
                file_name=c["file_name"],
                no_graphics=(c["file_name"] is not None),
                episode_horizon=c["episode_horizon"],
                timescale=timescale,
                high_level_steps=10,
                split_obs=use_split_obs,
            ),
        )

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("low_level_"):
                return "policy_low"
            else:
                return "policy_high"

        goal_vector_length = 10
        goal_vector_space = Box(0, 1, (goal_vector_length,))

        """
        List of observations:
        - (51) Vector obs:
            - (3) avg body vel relative to orientation cube (towards target)
            - (3) target direction
            - (4) rotation delta between body and target
            - (3) position of target relative to orientation cube (e.g. the agent)
            - (1) body raycast distance to ground
            - (17) body part information (e.g. touching ground, joint strength)
            - (20) Up/Down pointing raycasts
        - (126) Body part observations
            - ...
        
        """

        """
        Split vector obs:
        - (33,0) Upper policy: -> [0:13 + 31:52]
            - (3) Avg body vel
            - (3) target direction
            - (4) rotation delta
            - (3) relative position of target
            - (20) Up/Down raycasts
        - (38,126) Lower policy: -> [14:52]
            - (1) body raycast distance
            - (17) body part information
            - (20) Up/Down raycasts
            - (126) Body part observations
        
        """

        if use_split_obs:
            policies = {
                "policy_high": PolicySpec(
                    observation_space=TupleSpace(
                        [
                            Box(-np.inf, np.inf, (33,)),
                        ]
                    ),
                    action_space=goal_vector_space,  # Goal vector
                    # config={
                    #    "model": {
                    #        "custom_model": "HIROHigh",
                    #        "custom_model_config": {"fc_size": 512},
                    #    }
                    # },
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
                    # config={
                    #    "model": {
                    #        "custom_model": "HIROLow",
                    #        "custom_model_config": {"fc_size": 512, "goal_size": goal_vector_length},
                    #    }
                    # },
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
                    # config={
                    #    "model": {
                    #        "custom_model": "HIROHigh",
                    #        "custom_model_config": {"fc_size": 512},
                    #    }
                    # },
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
                    # config={
                    #    "model": {
                    #        "custom_model": "HIROLow",
                    #        "custom_model_config": {"fc_size": 512, "goal_size": goal_vector_length},
                    #    }
                    # },
                ),
            }

    else:  # Normal training without HRL
        from unity_env import BetterUnity3DEnv, HRLUnityEnv

        tune.register_env(
            "unity3d",
            lambda c: BetterUnity3DEnv(
                file_name=c["file_name"],
                no_graphics=(c["file_name"] is not None),
                episode_horizon=c["episode_horizon"],
                timescale=timescale,
            ),
        )

        # Get policies (different agent types; "behaviors" in MLAgents) and
        # the mappings from individual agents to Policies.
        policies, policy_mapping_fn = BetterUnity3DEnv.get_policy()

    enable_rl_module = True

    num_envs = args.num_workers if args.file_name else 1

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
        .rollouts(
            num_rollout_workers=args.num_workers if args.file_name else 0,
            rollout_fragment_length=int(args.horizon / 10),
            recreate_failed_workers=True,
        )
        .rl_module(_enable_rl_module_api=enable_rl_module)
        .training(
            lr=0.00015,  # [[0, 0.0003], [1_000_000 * 10 * 2, 0.0]], # env_steps * n_agents * (num_envs/2)
            lambda_=0.95,
            gamma=0.995,  # discount factor
            entropy_coeff=[
                [0, 0.0002],
                [1_000_000 * 10 * 2, 0.0],
            ],  # tune.grid_search([0.001, 0.0]),  # beta?
            sgd_minibatch_size=args.horizon * num_envs,  # batch_size?
            train_batch_size=args.horizon * num_envs,  # 20480  # buffer_size?
            num_sgd_iter=3,  # num_epoch?
            clip_param=0.3,  # epsilon?
            kl_target=0.01,
            kl_coeff=0.2,
            use_kl_loss=True,  # Must be True, use kl_coeff set to 0 to disable
            model={"fcnet_hiddens": [512, 512, 512], "fcnet_activation": "tanh"},
            _enable_learner_api=enable_rl_module,
        )
        .multi_agent(
            policies=policies, policy_mapping_fn=policy_mapping_fn, count_steps_by="env_steps"
        )  # Preferably use "env_steps" with HRL, because there are two different levels of policies, which messes up agent step count?
        .resources(
            num_gpus=args.gpus,
            num_gpus_per_worker=1 / args.num_workers,
        )
        # .resources(
        #    num_gpus=args.gpus,
        #    num_learner_workers=0,
        #    num_gpus_per_learner_worker=1,
        # )
    )

    print(len(config.to_dict()))

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.restore:
        tuner = tune.Tuner.restore(args.restore, "PPO")
    else:
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop=stop,
                verbose=3,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=5,
                    checkpoint_at_end=True,
                ),
                callbacks=[
                    WandbLoggerCallback(
                        project="HRL-Crawler",
                        group="GPU_PERF",
                    )
                ],
            ),
        )
    results = tuner.fit()

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
