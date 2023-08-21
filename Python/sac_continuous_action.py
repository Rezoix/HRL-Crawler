# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
#import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper 

from unity_env import BetterUnity3DEnv

from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=5,#default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=100,#default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args






def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        """ env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed) """
        #worker_id = 0
        #unity_env = UnityEnvironment(worker_id=worker_id, base_port=5004, timeout_wait=300) #TODO
        #env = UnityToGymWrapper(unity_env)
        env = BetterUnity3DEnv()
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forwards(self, obs, actions):
        """
        Takes multi-agent observations and actions in form of dict {agent_id: observations} and {agent_id: actions}
        """
        targets = {}
        for agent_id in obs:
            targets[agent_id] = self.forward(obs[agent_id], actions[agent_id])

        return targets


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_actions(self, obs):
        """
        Takes multi-agent replay buffer in form of dict {agent_id: buffer_entries}
        """
        next_actions = {}
        next_log_pi = {}
        next_means = {}
        for agent_id in obs:
            next_state_actions, next_state_log_pi, next_state_means = self.get_action(obs[agent_id])
            next_actions[agent_id] = next_state_actions
            next_log_pi[agent_id] = next_state_log_pi
            next_means[agent_id] = next_state_means

        return next_actions, next_log_pi, next_means

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # TODO: SyncVectorEnv doesnt work with multi-agent? 
    # Or only the RLlib method where observations are placed into dictionary with all agent names?

    # For now only use single environment instance at a time

    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)()

    max_action = float(env.action_space.high[0])

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    print(env.observation_space,
        env.action_space)
    
    print(type(env.observation_space),
        type(env.action_space))
    
    env.observation_space.dtype = np.float32
    """ rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False, #testing
    ) """

    rb = MultiAgentReplayBuffer(
        capacity=args.buffer_size,

    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs,_ = env.reset()

    # envs.reset() returns array of two arrays, with second one being empty. Why?
    # Work around it for now... It seems like it could be the 'infos'?
    #obs = obs[0]obs
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            #actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = env.action_space_sample()
        else:
            actions = {}
            for agent_id in obs:
                act, _, _ = actor.get_action(torch.Tensor([obs[agent_id]]).to(device))
                actions[agent_id] = act.detach().cpu().numpy()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
        dones = terminateds or truncateds
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        """ for agent_id in dones:
            done = dones[agent_id]
            if done:
                real_next_obs[agent_id] = infos[agent_id]["terminal_observation"] """
        
        sampleBatches = {}
        for agent_id in obs:
            sampleBatches[agent_id] = SampleBatch({
                "obs": torch.tensor([obs[agent_id]], device=device), 
                "new_obs": torch.tensor([real_next_obs[agent_id]], device=device), 
                "rewards": torch.tensor([rewards[agent_id]], device=device), 
                "actions": torch.tensor([actions[agent_id]], device=device),
                "terminateds": torch.tensor([terminateds[agent_id]], device=device),
                "truncateds": torch.tensor([truncateds[agent_id]], device=device)
                })
        multiBatches = MultiAgentBatch(sampleBatches, 1)
        rb.add(multiBatches)
        """ batch = SampleBatch({
            "obs": obs, 
            "new_obs": real_next_obs, 
            "rewards": rewards, 
            "actions": actions,
            "terminateds": terminateds,
            "truncateds": truncateds
        })

        print(batch["rewards"])

        rb.add(batch) """

        #rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to ovelog_alpharlook
        """ for agent_id in next_obs:
            obs[agent_id] = next_obs[agent_id] """
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                rb_new_obs = {k: v["new_obs"] for (k,v) in data.policy_batches.items()}
                rb_obs = {k: v["obs"] for (k,v) in data.policy_batches.items()}
                rb_act = {k: v["actions"] for (k,v) in data.policy_batches.items()}

                next_state_actions, next_state_log_pi, _ = actor.get_actions(rb_new_obs)
                qf1_next_target = qf1_target.forwards(rb_new_obs, next_state_actions)
                qf2_next_target = qf2_target.forwards(rb_new_obs, next_state_actions)
                #min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                min_qf_next_target = {}
                for agent_id in qf1_next_target:
                    min_qf_next_target[agent_id] = torch.min(qf1_next_target[agent_id], qf2_next_target[agent_id]) - alpha * next_state_log_pi[agent_id]
                
                #next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                next_q_value = {}
                for agent_id in data.policy_batches:
                    batch = data.policy_batches[agent_id]
                    re = batch["rewards"].flatten()
                    te = (1 - batch["terminateds"].flatten()) * args.gamma
                    mqf = (min_qf_next_target[agent_id]).view(-1)
                    next_q_value[agent_id] = re + te * mqf
                    #next_q_value[agent_id] = batch["rewards"].flatten() + (1 - batch["terminateds"].flatten()) * args.gamma * (min_qf_next_target[agent_id]).view(-1)

            qf1_a_values = {k: v.view(-1) for (k,v) in qf1.forwards(rb_obs, rb_act).items()}
            qf2_a_values = {k: v.view(-1) for (k,v) in qf2.forwards(rb_obs, rb_act).items()}
            qf1_loss = {k: F.mse_loss(qf1_a_values[k], next_q_value[k]) for k in next_q_value}
            qf2_loss = {k: F.mse_loss(qf2_a_values[k], next_q_value[k]) for k in next_q_value}
            qf_losses = [qf1_loss[k] + qf2_loss[k] for k in qf1_loss]
            qf_loss = 0
            for qfl in qf_losses:
                qf_loss += qfl
            qf_loss /= len(qf_losses)

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_actions(rb_obs)
                    qf1_pi = qf1.forwards(rb_obs, pi)
                    qf2_pi = qf2.forwards(rb_obs, pi)
                    #min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    min_qf_pi = {}
                    for agent_id in qf1_pi:
                        min_qf_pi[agent_id] = torch.min(qf1_pi[agent_id], qf2_pi[agent_id]).view(-1)
                    
                    #actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_loss = 0
                    for agent_id in min_qf_pi:
                        actor_loss += ((alpha * log_pi[agent_id]) - min_qf_pi[agent_id]).mean()
                    actor_loss /= len(min_qf_pi)

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_actions(rb_obs)
                        #alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                        alpha_loss = 0
                        for agent_id in log_pi:
                            alpha_loss += (-log_alpha * (log_pi[agent_id] +target_entropy)).mean()
                        alpha_loss /= len(log_pi)

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            """ if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step) """

    env.close()
    writer.close()
