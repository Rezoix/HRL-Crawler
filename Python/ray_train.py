import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved

from unity_env import BetterUnity3DEnv
from export import SaveCheckpointCallback

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=".\\models\\")
parser.add_argument("--restore", type=str, default=None, help="Filepath to checkpoint to restore training.")
parser.add_argument(
    "--file-name",
    type=str,
    default="..\\build\\Crawler.exe",
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
parser.add_argument("--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
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

    ray.init(num_gpus=args.gpus)

    timescale = 1.25

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
            rollout_fragment_length=200,
        )
        .rl_module(_enable_rl_module_api=enable_rl_module)
        .training(
            lr=0.0003,
            lambda_=0.95,
            gamma=0.995,  # discount factor
            entropy_coeff=0.005,  # beta?
            sgd_minibatch_size=2048,  # batch_size?
            train_batch_size=2048 * 8,  # 20480  # buffer_size?
            num_sgd_iter=3,  # num_epoch?
            clip_param=0.2,  # epsilon?
            model={"fcnet_hiddens": [512, 512, 512]},
            _enable_learner_api=enable_rl_module,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            count_steps_by="env_steps",  # "agent_steps"?
        )
        .resources(
            num_gpus=args.gpus,
            num_gpus_per_worker=1 / (args.num_workers if args.num_workers > 0 else 1),
        )
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.restore:
        tuner = tune.Tuner.restore(args.restore)
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
                    SaveCheckpointCallback(
                        ["Crawler"],
                        args.model_path,
                    )
                ],
            ),
        )

    results = tuner.fit()

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
