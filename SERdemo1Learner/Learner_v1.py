import RLbrain_v1
from RLbrain_v1 import Agent

import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


device = torch.device('cuda')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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


def send_parameters():
    # socket send parameters back to worker
    # when receive newest parameter request, call this
    return


def receive_exp(learner):
    # socket receive experience, then perform training
    # when receive experience from workers, call this
    RLbrain_v1.training_process(learner)
    return


def training_process(leaner):
    return


if __name__ == "__main__":
    learner = Agent(num_actions=6)

    memory = ReplayMemory(10000)

    optimizer = optim.RMSprop(learner.parameters())
    memory = ReplayMemory(10000)
    training_process(learner)
    # socket listen


