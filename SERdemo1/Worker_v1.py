import time
import experiment_api
import RLbrain_v1
from RLbrain_v1 import Agent

import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import random


# if gpu is to be used
device = torch.device('cuda')

Transition = namedtuple('Transition', ('state', 'action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def pull_parameters():
    # socket get learner side .state_dict()
    return


def send_exp():
    # socket send experience to learner
    return


def compute_reward(old_state, new_state, init_state):
    # compare state, if position changed get reward. if z axis is lower than initial, get big reward.

    x_new = new_state[1].position.x
    y_new = new_state[1].position.y
    z_new = new_state[1].position.z

    new_position_np = np.array((x_new, y_new, z_new))

    x_old = old_state[1].position.x
    y_old = old_state[1].position.y
    z_old = old_state[1].position.z

    old_position_np = np.array((x_old, y_old, z_old))

    distance = np.linalg.norm(new_position_np - old_position_np)

    z_init = init_state[1].position.z
    # check if blue object fell from the table
    eps = 0.1
    if z_new + eps < z_init:
        return 100

    # check if blue object was moved
    if(distance > eps):
        return 10

    return 1


def select_strategy(strategy_threshold):
    prob = random.uniform(0, 1)
    strategy = 'exploit'
    if prob < strategy_threshold:
        strategy = 'explore'
    return strategy


def check_done(new_state, init_state):
    # + 0.1 because there are some noises
    if((new_state[1].position.z + 0.1) < init_state[1].position.z):
        return True
    else:
        return False


robot = experiment_api.Robot()
time.sleep(1)
worker = Agent(num_actions=6)


memory = ReplayMemory(10000)

num_episodes = 50
for i in range(num_episodes):
    robot.reset()
    init_state = robot.get_current_state()
    state = init_state
    strategy_threshold = 1 - 1/(num_episodes - i)
    strategy = select_strategy(strategy_threshold)
    for actions_counter in count():
        if strategy == 'exploit':
            action = worker(state)
        else:
            action = []
            for i in range(6):
                action.append(random.uniform(-3, 3))
        j1, j2, j3, j4, j5, j6 = action
        robot.act(j1, j2, j3, j4, j5, j6)
        time.sleep(1)
        new_state = robot.get_current_state()
        reward = compute_reward(state, new_state, init_state)

        memory.push(state, action, reward)

        state = new_state

        if check_done(init_state, new_state):
            break

    send_exp()  # one game over, send the experience
    print('number of actions in this round game:', actions_counter)

print(num_episodes, ' training over')
