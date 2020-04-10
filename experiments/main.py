"""Entry point for training, transferring and visualising agents."""
import numpy as np
from itertools import count
from collections import namedtuple
import argparse
from time import time, sleep
from pprint import pprint
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wurm.envs import SingleSnake, SimpleGridworld
from wurm import agents
from wurm.utils import env_consistency, CSVLogger, ExponentialMovingAverageTracker
from wurm.rl import A2C, TrajectoryStore
from config import BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL, PATH


RENDER = False
LOG_INTERVAL = 100
MAX_GRAD_NORM = 0.5
FPS = 10


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--num-envs', type=int)
parser.add_argument('--size', type=int)
parser.add_argument('--agent', type=str)
parser.add_argument('--train', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--observation', default='default', type=str)
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--render', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--render-window-size', default=256, type=int)
parser.add_argument('--render-cols', default=1, type=int)
parser.add_argument('--render-rows', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--update-steps', default=20, type=int)
parser.add_argument('--entropy', default=0.0, type=float)
parser.add_argument('--total-steps', default=float('inf'), type=float)
parser.add_argument('--total-episodes', default=float('inf'), type=float)
parser.add_argument('--save-location', type=str, default=None)
parser.add_argument('--save-model', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-logs', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-video', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--r', default=None, type=int, help='Repeat number')
args = parser.parse_args()

excluded_args = ['train', 'device', 'verbose', 'save_location', 'save_model', 'save_logs', 'render',
                 'render_window_size', 'render_rows', 'render_cols', 'save_video']
if args.r is None:
    excluded_args += ['r', ]
if args.total_steps == float('inf'):
    excluded_args += ['total_steps']
if args.total_episodes == float('inf'):
    excluded_args += ['total_episodes']
argsdict = {k: v for k, v in args.__dict__.items() if k not in excluded_args}
argstring = '__'.join([f'{k}={v}' for k, v in argsdict.items()])
print(argstring)


Transition = namedtuple('Transition', ['action', 'log_prob', 'value', 'reward', 'done', 'entropy'])

# Save file
if args.save_location is None:
    save_file = argstring
else:
    save_file = args.save_location


###################
# Configure Agent #
###################
# Reload/fresh model
if os.path.exists(args.agent) or os.path.exists(os.path.join(PATH, 'models', args.agent)):
    # Agent is to be loaded from a file
    agent_path = args.agent if os.path.exists(args.agent) else os.path.join(PATH, 'models', args.agent)
    agent_str = args.agent.split('/')[-1][:-3]
    agent_params = {kv.split('=')[0]: kv.split('=')[1] for kv in agent_str.split('__')}
    agent_type = agent_params['agent']
    observation_type = agent_params['observation']
    reload = True
    print('Loading agent from file. Agent params:')
    pprint(agent_params)
else:
    # Creating a new agent
    agent_type = args.agent
    observation_type = args.observation
    reload = False

# Configure observation type
if observation_type == 'one_channel':
    in_channels = 1
elif observation_type == 'default':
    in_channels = 3
elif observation_type == 'raw':
    if args.env == 'gridworld':
        in_channels = 2
    elif args.env == 'snake':
        in_channels = 3
    else:
        raise RuntimeError
elif observation_type.startswith('partial_'):
    in_channels = 3
else:
    raise ValueError

# Create agent
if agent_type == 'relational':
    model = agents.RelationalAgent(num_actions=4, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                                   num_relational=2, num_attention_heads=2, relational_dim=32, num_feedforward=1,
                                   feedforward_dim=64, residual=True).to(args.device)
elif agent_type == 'simpleconv':
    model = agents.SimpleConvAgent(
        in_channels=in_channels, size=args.size, coord_conv=args.coord_conv).to(
        args.device)
elif agent_type == 'convolutional':
    model = agents.ConvAgent(num_actions=4, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                             num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(args.device)
elif agent_type == 'feedforward':
    if observation_type == 'positions':
        num_inputs = 4
    elif observation_type.startswith('partial_'):
        observation_size = int(observation_type.split('_')[-1])
        observation_width = 2 * observation_size + 1
        num_inputs = 3 * (observation_width ** 2)
    else:
        raise ValueError('Feedforward agent only compatible with partial and position observations.')

    model = agents.FeedforwardAgent(num_actions=4, num_inputs=num_inputs, num_layers=2, hidden_units=64).to(args.device)
elif agent_type == 'random':
    model = agents.RandomAgent(num_actions=4, device=args.device)
else:
    raise ValueError('Unrecognised agent')

if reload:
    model.load_state_dict(torch.load(agent_path))

if args.train:
    model.train()
else:
    torch.no_grad()
    model.eval()
print(model)

if args.agent != 'random' and args.train:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()


#################
# Configure Env #
#################
render_args = {
    'size': args.render_window_size,
    'num_rows': args.render_rows,
    'num_cols': args.render_cols,
}
if args.env == 'gridworld':
    env = SimpleGridworld(num_envs=args.num_envs, size=args.size, start_location=(args.size//2, args.size//2),
                          observation_mode=observation_type, device=args.device)
elif args.env == 'snake':
    env = SingleSnake(num_envs=args.num_envs, size=args.size, device=args.device,
                      observation_mode=observation_type, render_args=render_args)
else:
    raise ValueError('Unrecognised environment')


trajectories = TrajectoryStore()
ewm_tracker = ExponentialMovingAverageTracker(alpha=0.025)

episode_length = 0
num_episodes = 0
num_steps = 0
if args.save_logs:
    logger = CSVLogger(filename=f'{PATH}/logs/{save_file}.csv')
if args.save_video:
    os.makedirs(PATH + f'/videos/{save_file}', exist_ok=True)
    recorder = VideoRecorder(env, path=PATH + f'/videos/{save_file}/0.mp4')

a2c = A2C(model, gamma=args.gamma)


############################
# Run agent in environment #
############################
t0 = time()
state = env.reset()
for i_step in count(1):
    if args.render:
        env.render()
        sleep(1. / FPS)

    if args.save_video:
        recorder.capture_frame()

    #############################
    # Interact with environment #
    #############################
    probs, state_value = model(state)
    action_distribution = Categorical(probs)
    entropy = action_distribution.entropy().mean()
    action = action_distribution.sample().clone().long()

    state, reward, done, info = env.step(action)

    if args.env == 'snake':
        env_consistency(env.envs[~done.squeeze(-1)])

    if args.agent != 'random' and args.train:
        trajectories.append(
            action=action,
            log_prob=action_distribution.log_prob(action),
            value=state_value,
            reward=reward,
            done=done,
            entropy=entropy
        )

    env.reset(done)

    ##########################
    # Advantage actor-critic #
    ##########################
    if args.agent != 'random' and args.train and i_step % args.update_steps == 0:
        with torch.no_grad():
            _, bootstrap_values = model(state)

        value_loss, policy_loss = a2c.loss(bootstrap_values, trajectories.rewards, trajectories.values,
                                           trajectories.log_probs, trajectories.dones)

        entropy_loss = - trajectories.entropies.mean()

        optimizer.zero_grad()
        loss = value_loss + policy_loss + args.entropy * entropy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        trajectories.clear()

    ###########
    # Logging #
    ###########
    num_episodes += done.sum().item()
    num_steps += args.num_envs

    if args.save_video:
        # If there is just one env save each episode to a different file
        # Otherwise save the whole video at the end
        if args.num_envs == 1:
            if done:
                # Save video and make a new recorder
                recorder.close()
                recorder = VideoRecorder(env, path=PATH + f'/videos/{save_file}/{num_episodes}.mp4')

    ewm_tracker(
        reward_rate=reward.mean().item(),
        entropy=entropy.mean().item(),
        edge_collisions=info['edge_collision'].float().mean().item()
    )

    if args.env == 'snake':
        ewm_tracker(
            self_collisions=info['self_collision'].float().mean().item(),
            avg_size=env.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(args.num_envs, -1).max(dim=1)[0].mean().item()
        )

    if i_step % LOG_INTERVAL == 0:
        if args.save_model:
            os.makedirs(f'{PATH}/models/', exist_ok=True)
            torch.save(model.state_dict(), f'{PATH}/models/{save_file}.pt')

        # Logging
        t = time() - t0
        log_string = '[{:02d}:{:02d}:{:02d}]\t'.format(int((t // 3600)), int((t // 60) % 60), int(t % 60))
        log_string += 'Steps {:.2f}e6\t'.format(num_steps/1e6)
        log_string += 'Reward rate: {:.3e}\t'.format(ewm_tracker['reward_rate'])
        log_string += 'Entropy: {:.3e}\t'.format(ewm_tracker['entropy'])

        logs = {
            't': t,
            'steps': num_steps,
            'episodes': num_episodes,
            'reward_rate': reward.mean().item(),
            'policy_entropy': entropy.mean().item(),
            'edge_collisions': info['edge_collision'].float().mean().item()
        }
        if args.agent != 'random' and args.train:
            logs.update({
                'value_loss': value_loss,
                'policy_loss': policy_loss,
                'entropy_loss': entropy_loss,
            })

        log_string += 'Edge Collision: {:.3e}\t'.format(ewm_tracker['edge_collisions'])

        if args.env == 'snake':
            log_string += 'Avg. size: {:.3f}\t'.format(ewm_tracker['avg_size'])
            log_string += 'Self Collision: {:.3e}\t'.format(ewm_tracker['self_collisions'])
            logs.update({
                'avg_size': env.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(args.num_envs, -1).max(dim=1)[0].mean().item(),
                'self_collisions': info['self_collision'].float().mean().item()
            })

        if args.save_logs:
            logger.write(logs)

        ewm_tracker(**logs)

        print(log_string)

    if num_steps > args.total_steps or num_episodes > args.total_episodes:
        break

if args.save_video:
    recorder.close()
