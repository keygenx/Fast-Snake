from time import time
from collections import namedtuple
import torch
from torch.nn import functional as F
from typing import Tuple
from gym.envs.classic_control import rendering
from PIL import Image
import os
import numpy as np

from wurm._filters import ORIENTATION_FILTERS, NO_CHANGE_FILTER
from wurm.utils import head, food, body, drop_duplicates

################################CONSTANT#########################################
#PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_DEVICE = 'cuda'

FOOD_CHANNEL = 0
HEAD_CHANNEL = 1

EDGE_COLLISION_REWARD = -1

STEP_REWARD = -0.1
FOOD_REWARD = +0.5

EPS = 1e-6
################################CONSTANT#########################################


Spec = namedtuple('Spec', ['reward_threshold'])

class SimpleGridworld(object):
    """Batched gridworld environment.

    In this environment the agent can move in the 4 cardinal directions and receives +1 reward when moving on to a food
    square. At which point either the episode is finished or the food respawns. Moving off the edge of the gridworld
    results in a death.

    Each environment is represented as a Tensor image with 2 channels. Each channel has the following meaning:
    0 - Food channel. 1 = food, 0 = no food
    1 - Agent channel. 1 = agent location, 0 = empty

    Example of a single environment containing a single agent and a food object.

    Food channel                 Agent channel
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   | 1 |   |   |   |   |    |   |   |   |   | 1 |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    """

    spec = Spec(float('inf'))
    metadata = {
    'render.modes': ['rgb_array'],
    'video.frames_per_second': 12
    }

    def __init__(self,
                 num_envs: int,
                 size: int,
                 auto_reset: bool = True ,
                 observation_mode: str = 'default',
                 device: str = DEFAULT_DEVICE,
                 start_location: Tuple[int, int] = None,
                 manual_setup: bool = False,
                 render_args = None,
                 verbose: int = 0):
        """Initialise the environments

        Args:
            num_envs:
            size:
            on_death:
        """
        self.num_envs = num_envs
        self.size = size
        self.observation_mode = observation_mode
        self.start_location = start_location
        self.device = device
        self.auto_reset = auto_reset
        self.verbose = verbose
        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 512}
        else:
            self.render_args = render_args
            
        self.t = 0

        if manual_setup:
            # All zeros, user must create environment
            self.envs = torch.zeros((num_envs, 2, size, size)).to(self.device).requires_grad_(False)
        else:
            # Create environments automatically
            self.envs = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        self.done = torch.zeros(num_envs).to(self.device).bool()

        self.viewer = None

        self.head_colour = torch.Tensor((0, 255, 0)).short().to(self.device)
        self.food_colour = torch.Tensor((255, 0, 0)).short().to(self.device)
        self.edge_colour = torch.Tensor((0, 0, 0)).short().to(self.device)
        
        self.move_left = torch.Tensor([1]).long().to(self.device)
        self.move_up = torch.Tensor([2]).long().to(self.device)
        self.move_right = torch.Tensor([3]).long().to(self.device)
        self.move_down = torch.Tensor([0]).long().to(self.device)
        self.action_space = torch.Tensor([self.move_left, self.move_up, self.move_right, self.move_down]).long().to(self.device)
    
    
    def _get_rgb(self):
        # RGB image same as is displayed in .render()
        img = torch.ones((self.num_envs, 3, self.size, self.size)).short().to(self.device) * 255
        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        head_locations = (head(self.envs) > EPS).squeeze(1)
        img[head_locations, :] = self.head_colour

        food_locations = (food(self.envs) > EPS).squeeze(1)
        img[food_locations, :] = self.food_colour

        img[:, :1, :, :] = self.edge_colour
        img[:, :, :1, :] = self.edge_colour
        img[:, -1:, :, :] = self.edge_colour
        img[:, :, -1:, :] = self.edge_colour

        # Convert back to BCHW axes
        img = img.permute((0, 3, 1, 2))

        return img

    def _observe(self, observation_mode: str = 'default'):
        if observation_mode == 'default':
            # RGB image same as is displayed in .render()
            observation = self._get_rgb()

            # Normalise to 0-1
            observation = observation.float() / 255

            return observation
        elif observation_mode == 'raw':
            return self.envs.clone()
        
        elif observation_mode == 'one_channel':
            observation = (self.envs[:, HEAD_CHANNEL, :, :] == 1) * 1.0
            observation += self.envs[:, FOOD_CHANNEL, :, :] * 3.0
            # Add in -1 values to indicate edge of map
            observation[:, :1, :] = -1
            observation[:, :, :1] = -1
            observation[:, -1:, :] = -1
            observation[:, :, -1:] = -1
            return observation.unsqueeze(1)
        
        elif observation_mode == 'positions':
            head_idx = self.envs[:, HEAD_CHANNEL, :, :].view(self.num_envs, self.size ** 2).argmax(dim=-1)
            food_idx = self.envs[:, FOOD_CHANNEL, :, :].view(self.num_envs, self.size ** 2).argmax(dim=-1)
            observation = torch.Tensor([
                head_idx // self.size,
                head_idx % self.size,
                food_idx // self.size,
                food_idx % self.size
            ]).float().unsqueeze(0)
            return observation
        else:
            raise Exception

    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        if actions.dtype not in (torch.short, torch.int, torch.long):
            raise TypeError('actions Tensor must be an integer type i.e. '
                            '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

        if actions.shape[0] != self.num_envs:
            raise RuntimeError('Must have the same number of actions as environments.')

        reward = torch.zeros((self.num_envs,)).float().to(self.device).requires_grad_(False)
        previous_done = self.done.clone() #
        info = dict()

        t0 = time()
        # Create head position deltas
        head_deltas = F.conv2d(head(self.envs), ORIENTATION_FILTERS.to(self.device), padding=1)
        # Select the head position delta corresponding to the correct action
        actions_onehot = torch.FloatTensor(self.num_envs, 4).to(self.device)
        actions_onehot.zero_()
        actions_onehot.scatter_(1, actions.unsqueeze(-1), 1)
        head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, actions_onehot]).unsqueeze(1)

        # Move head position by applying delta
        self.envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, :, :].add_(head_deltas).round_()
        if self.verbose:
            print(f'Head movement: {time() - t0}s')

        ################
        # Apply update #
        ################

        t0 = time()
        # Remove food and give reward
        # `food_removal` is 0 except where a snake head is at the same location as food where it is -1
        food_removal = head(self.envs) * food(self.envs) * -1
        reward.sub_(FOOD_REWARD, food_removal.view(self.num_envs, -1).sum(dim=-1).float())
        self.envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_removal
        if self.verbose:
            print(f'Food removal: {time() - t0}s')

        # Add new food if necessary.
        if food_removal.sum() < 0:
            t0 = time()
            food_addition_env_indices = (food_removal * -1).view(self.num_envs, -1).sum(dim=-1).bool()
            add_food_envs = self.envs[food_addition_env_indices, :, :, :]
            food_addition = self._get_food_addition(add_food_envs)
            self.envs[food_addition_env_indices, FOOD_CHANNEL:FOOD_CHANNEL+1, :, :] += food_addition
            if self.verbose:
                print(f'Food addition ({food_addition_env_indices.sum().item()} envs): {time() - t0}s')

        t0 = time()
        # Check for boundary, Done by performing a convolution with no padding
        # If the head is at the edge then it will be cut off and the sum of the head
        # channel will be 0
        edge_collision = F.conv2d(
            head(self.envs),
            NO_CHANGE_FILTER.to(self.device),
        ).view(self.num_envs, -1).sum(dim=-1) < EPS
        self.done = self.done | edge_collision
        reward.add_(EDGE_COLLISION_REWARD, edge_collision)
        info.update({'edge_collision': edge_collision})
        if self.verbose:
            print(f'Edge collision ({edge_collision.sum().item()} envs): {time() - t0}s')

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()

        
        #Applying step reward
        reward.add_(STEP_REWARD)
        if ~self.auto_reset:
            reward.mul_(~previous_done)

        done = self.done.clone()

        if self.done.any() and self.auto_reset:
            self.reset(self.done)
        return self._observe(self.observation_mode), reward, done, info

    def _select_from_available_locations(self, locs: torch.Tensor) -> torch.Tensor:
        locations = torch.nonzero(locs)
        random_loc = locations[torch.randperm(locations.shape[0])[:1]]
        return random_loc

    def _get_food_addition(self, envs: torch.Tensor):
        # Get empty locations
        available_locations = envs.sum(dim=1, keepdim=True) < EPS
        # Remove boundaries
        available_locations[:, :, :1, :] = 0
        available_locations[:, :, :, :1] = 0
        available_locations[:, :, -1:, :] = 0
        available_locations[:, :, :, -1:] = 0

        food_indices = drop_duplicates(torch.nonzero(available_locations), 0)
        food_addition = torch.sparse_coo_tensor(
            food_indices.t(),  torch.ones(len(food_indices)), available_locations.shape, device=self.device)
        food_addition = food_addition.to_dense()

        return food_addition

    def reset(self, done: torch.Tensor = None):
        """Resets environments in which the snake has died

        Args:
            done: A 1D Tensor of length self.num_envs. A value of 1 means the corresponding environment needs to be
                reset
        """
        if done is None:
            done = torch.ones(self.num_envs, dtype=bool).to(self.device)


        t0 = time()
        if done.sum() > 0:
            new_envs = self._create_envs(int(done.sum().item()))
            self.envs[done, :, :, :] = new_envs

        if self.verbose:
            print(f'Resetting {done.sum().item()} envs: {time() - t0}s')
        self.done = self.done & (~done)
        return self._observe(self.observation_mode)

    def _create_envs(self, num_envs: int):
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.size <= 4:
            raise NotImplementedError('Environemnts smaller than this don\'t make sense.')

        envs = torch.zeros((num_envs, 2, self.size, self.size)).to(self.device)
        
        if self.start_location is None:
            available_locations = envs.sum(dim=1, keepdim=True) < EPS
            # Remove boundaries
            available_locations[:, :, :1, :] = 0
            available_locations[:, :, :, :1] = 0
            available_locations[:, :, -1:, :] = 0
            available_locations[:, :, :, -1:] = 0

            head_indices = drop_duplicates(torch.nonzero(available_locations), 0)
            head_addition = torch.sparse_coo_tensor(
                head_indices.t(),  torch.ones(len(head_indices)), available_locations.shape, device=self.device)
            envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, :, :] = head_addition.to_dense()
        
        else:
            envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, self.start_location[0], self.start_location[1]] = 1

        # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        return envs.round()

    def render(self, mode: str = 'human'):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        # Get RBG Tensor BCHW
        img = self._get_rgb()

        # Convert to numpy
        img = img.cpu().numpy()
        # Rearrange images depending on number of envs
        if self.num_envs == 1:
            num_cols = num_rows = 1
            img = img[0]
            img = np.transpose(img, (1, 2, 0))
        else:
            num_rows = self.render_args['num_rows']
            num_cols = self.render_args['num_cols']
            # Make a 2x2 grid of images
            output = np.zeros((self.size*num_rows, self.size*num_cols, 3))
            for i in range(num_rows):
                for j in range(num_cols):
                    output[
                        i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size, :
                    ] = np.transpose(img[i*num_cols + j], (1, 2, 0))

            img = output
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize(
            (self.render_args['size'] * num_cols,
             self.render_args['size'] * num_rows), Image.NEAREST
        ))

        if mode == 'human':
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError('Render mode not recognised.')

    def close(self):
        self.viewer.window.close()
        self.viewer= None
        
    def random_action(self):
        return self.action_space[torch.randint(0,4,(self.num_envs,))]
    
    def _consistent(self):
        pass
