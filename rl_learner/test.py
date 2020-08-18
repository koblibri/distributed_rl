from collections import namedtuple
import random
import torch
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


memory = ReplayMemory(10000)

state1 = torch.FloatTensor([1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
action1 = torch.FloatTensor([0,0,0,0,0,1])
reward1 = torch.FloatTensor([0])
state2 = torch.FloatTensor([1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
action2 = torch.FloatTensor([0,1,0,0,0,0])
reward2 = torch.FloatTensor([1])

memory.push(state1, action1, reward1)
memory.push(state2, action2, reward2)
transitions = memory.sample(128)
# batch = Transition(*zip(*transitions))
# print(batch)
# state_batch = torch.stack(batch.state, dim=0)
# action_batch = torch.stack(batch.action, dim=0)
# reward_batch = torch.stack(batch.reward, dim=0)
# print(state_batch)
# print(action_batch)
# print(reward_batch)