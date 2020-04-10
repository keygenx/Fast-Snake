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

from wurm.envs import SingleSnake
from config import BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL, PATH

###############
class AddCoords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret


class CoordConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.addcoords = AddCoords()
        in_size = in_channels+2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class MultiHeadDotProductAttention(nn.Module):

    def __init__(self, num_heads: int, input_dim: int, output_dim: int):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError('Number of num_heads must divide')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.per_head_dim = output_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.out = nn.Linear(output_dim, output_dim)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.per_head_dim, elementwise_affine=False)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        return output

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Calculate queries, keys, values and split into num_heads
        k = self.layer_norm(self.k_linear(x).view(batch_size, -1, self.num_heads, self.per_head_dim))
        q = self.layer_norm(self.q_linear(x).view(batch_size, -1, self.num_heads, self.per_head_dim))
        v = self.layer_norm(self.v_linear(x).view(batch_size, -1, self.num_heads, self.per_head_dim))

        # Transpose to get dimensions batch_size * num_heads * sequence_length * input_dim
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.per_head_dim)

        # Concatenate num_heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.output_dim)

        output = self.out(concat)

        return output


class RelationalModule2D(nn.Module):
    """Implements the relational module from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 output_dim: int,
                 residual: bool,
                 add_coords: bool = True):
        super().__init__()
        if add_coords:
            self.addcoords = AddCoords()
            input_dim = input_dim + 2
        self.attention = MultiHeadDotProductAttention(num_heads, input_dim, output_dim)
        self.residual = residual

    def forward(self, x: torch.Tensor):
        identity = x
        n, c, h, w = x.size()

        if hasattr(self, 'addcoords'):
            x = self.addcoords(x)
            c += 2

        # Unroll the 2D image tensor to a sequence so it can be fed to
        # the attention module then return to original shape
        out = x.view(n, c, h*w).transpose(1, 2)  # n, h*w, c
        out = self.attention(out)
        out = out.transpose(2, 1).view(n, self.attention.output_dim, h, w)

        if self.residual:
            out += identity

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool, add_coords: bool = True):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if residual:
            assert in_channels == out_channels
        self.conv = CoordConv2D(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv(x)
        out = F.relu(out)

        if self.residual:
            out += identity

        return out


def feedforward_block(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU()
    )
#########################################################################
class ConvAgent(nn.Module):
    """Implementation of baseline agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 in_channels: int,
                 num_initial_convs: int,
                 num_residual_convs: int,
                 num_feedforward: int,
                 feedforward_dim: int,
                 num_actions: int,
                 conv_channels: int = 16,
                 num_heads: int = 1):
        super(ConvAgent, self).__init__()
        self.in_channels = in_channels
        self.num_initial_convs = num_initial_convs
        self.num_residual_convs = num_residual_convs
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.conv_channels = conv_channels
        self.num_actions = num_actions
        self.num_heads = num_heads

        initial_convs = [ConvBlock(self.in_channels, self.conv_channels, residual=False), ]
        for _ in range(self.num_initial_convs - 1):
            initial_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=False))

        self.initial_conv_blocks = nn.Sequential(*initial_convs)

        residual_convs = [ConvBlock(self.conv_channels, self.conv_channels, residual=True), ]
        for _ in range(self.num_residual_convs - 1):
            residual_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=True))

        self.residual_conv_blocks = nn.Sequential(*residual_convs)

        feedforwards = [feedforward_block(self.conv_channels, self.feedforward_dim), ]
        for _ in range(self.num_feedforward - 1):
            feedforwards.append(feedforward_block(self.feedforward_dim, self.feedforward_dim))

        self.feedforward = nn.Sequential(*feedforwards)

        self.value_head = nn.Linear(self.feedforward_dim, num_heads)
        self.policy_head = nn.Linear(self.feedforward_dim, self.num_actions * num_heads)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x)
        x = self.residual_conv_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)
        values = self.value_head(x)
        action_probabilities = self.policy_head(x)
        return F.softmax(action_probabilities, dim=-1), values

#################################################################################

from typing import Callable

class A2C():
    """Class that encapsulates the advantage actor-critic algorithm.

    Args:
        actor_critic: Module that outputs
        gamma: Discount value
        value_loss_fn: Loss function between values and returns i.e. Huber, MSE
        normalise_returns: Whether or not to normalise target returns
    """
    def __init__(self,
                 gamma: float = 0.99,
                 value_loss_fn: Callable = F.smooth_l1_loss,
                 normalise_returns: bool = False,
                 use_gae: bool = False,
                 gae_lambda: float = None,
                 dtype: torch.dtype = torch.float):

        self.gamma = gamma
        self.normalise_returns = normalise_returns
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.value_loss_fn = value_loss_fn
        self.dtype = dtype

    def loss(self,
             bootstrap_values: torch.Tensor,
             rewards: torch.Tensor,
             values: torch.Tensor,
             log_probs: torch.Tensor,
             dones: torch.Tensor,
             return_returns: bool = False):
        """Calculate A2C loss.

        Args:
            bootstrap_values: Vector containing estimated value of final states each trajectory.
                Shape (num_envs, 1)
            rewards: Rewards for trajectories. Shape: (num_envs, num_steps)
            values: Values for trajectory states: Shape (num_envs, num_steps)
            log_probs: Log probabilities of actions taken during trajectory. Shape: (num_envs, num_steps)
            dones: Done masks for trajectory states. Shape: (num_envs, num_steps)
        """
        returns = []
        if self.use_gae:
            gae = 0
            for t in reversed(range(rewards.size(0))):
                if t == rewards.size(0) - 1:
                    delta = rewards[t] + self.gamma * bootstrap_values * (~dones[t]).to(self.dtype) - values[t]
                else:
                    delta = rewards[t] + self.gamma * values[t+1] * (~dones[t]).to(self.dtype) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (~dones[t]).to(self.dtype) * gae
                R = gae + values[t]
                returns.insert(0, R)
    
        else:
            R = bootstrap_values * (~dones[-1]).to(self.dtype)
            for r, d in zip(reversed(rewards), reversed(dones.int())):
                R = r + self.gamma * R * (~d).to(self.dtype)

                returns.insert(0, R)

        returns = torch.stack(returns)
        if self.normalise_returns:
            returns = (returns - returns.mean()) / (returns.std() + EPS)

        value_loss = self.value_loss_fn(values, returns).mean()
        advantages = returns - values
        policy_loss = - (advantages.squeeze(-1).detach() * log_probs).mean()

        ret = (value_loss, policy_loss)
        if return_returns:
            ret += returns

        return ret
    
    def print_vars(self):
        print("Gamma:", self.gamma)
 ####3######################################################

class TrajectoryStore(object):
    """Stores list of transitions.

    Each property should return a tensor of shape (num_steps, num_envs, 1)
    """
    def __init__(self):
        self.clear()

    def append(self,
               state: torch.Tensor = None,
               action: torch.Tensor = None,
               log_prob: torch.Tensor = None,
               reward: torch.Tensor = None,
               value: torch.Tensor = None,
               done: torch.Tensor = None,
               entropy: torch.Tensor = None,
               hidden_state: torch.Tensor = None):
        """Adds a transition to the store.

        Each argument should be a vector of shape (num_envs, 1)
        """
        if state is not None:
            self._states.append(state)

        if action is not None:
            self._actions.append(action)

        if log_prob is not None:
            self._log_probs.append(log_prob)

        if reward is not None:
            self._rewards.append(reward)

        if value is not None:
            self._values.append(value)

        if done is not None:
            self._dones.append(done)

        if entropy is not None:
            self._entropies.append(entropy)

        if hidden_state is not None:
            self._hiddens.append(hidden_state)

    def clear(self):
        self._states = []
        self._actions = []
        self._log_probs = []
        self._rewards = []
        self._values = []
        self._dones = []
        self._entropies = []
        self._hiddens = []

    @property
    def states(self):
        return torch.stack(self._states)

    @property
    def actions(self):
        return torch.stack(self._actions)

    @property
    def log_probs(self):
        return torch.stack(self._log_probs)

    @property
    def rewards(self):
        return torch.stack(self._rewards)

    @property
    def values(self):
        return torch.stack(self._values)

    @property
    def dones(self):
        return torch.stack(self._dones)

    @property
    def entropies(self):
        return torch.stack(self._entropies)

    @property
    def hidden_state(self):
        return torch.stack(self._hiddens)




env = SingleSnake(num_envs=512, size=10, observation_mode='raw')
#env.render()
model = ConvAgent(num_actions=4, num_initial_convs=2, in_channels=3, conv_channels=32,
                             num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

trajectories = TrajectoryStore()

a2c = A2C(gamma = 0.99)

a2c.print_vars()
#torch.no_grad()
#model.eval()
print("yay")



render = False
MAX_GRAD_NORM = 0.5
args_entropy = 0.01
update_steps = 50
total_steps = 1000
############################
# Run agent in environment #
############################
t0 = time()
state = env.reset()

for i_step in count(1):

    #print("step:",i_step, end='\r')
    if render:
        env.render()
        sleep(1. / FPS)

    #############################
    # Interact with environment #
    #############################
    
    probs, state_value = model(state)
    
    action_distribution = Categorical(probs)
    
    entropy = action_distribution.entropy().mean()
    action = action_distribution.sample().clone().long()
    print(i_step)
    state, reward, done, info = env.step(action)
    trajectories.append(
        action=action,
        log_prob=action_distribution.log_prob(action),
        value=state_value,
        reward=reward,
        done=done,
        entropy=entropy
    )
    #print(state)

    
    env.reset(done)

    ##########################
    # Advantage actor-critic #
    ##########################
    if i_step % update_steps == 0:
        with torch.no_grad():
            _, bootstrap_values = model(state)

        value_loss, policy_loss = a2c.loss(bootstrap_values, trajectories.rewards, trajectories.values,
                                           trajectories.log_probs, trajectories.dones)

        entropy_loss = - trajectories.entropies.mean()

        optimizer.zero_grad()
        loss = value_loss + policy_loss + args_entropy * entropy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        trajectories.clear()

    ###########
    if i_step == total_steps:
        break
if render:
    env.close()