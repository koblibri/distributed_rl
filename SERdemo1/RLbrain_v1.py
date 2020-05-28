import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.l1 = nn.Linear(9, 128)  # 7 input state, how many? three, only x y z??
        self.l2 = nn.Linear(128, num_actions)

    def forward(self, x): 
        # change tuple to array, add position of object
        parameters = list(x[0]) + [x[1].position.x, x[1].position.y, x[1].position.z]
        x = torch.tensor(parameters)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        #linear activation no extra function necessary
        return x

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_actions = 6

    def forward(self, q_pred, true_action, discounted_reward):
        # define the loss,
        one_hot = torch.zeros(len(true_action), self.num_actions).scatter_(1, true_action, 1)
        neg_log_prob = torch.sum(-torch.log(self.q_pred) * one_hot, dim=1)
        loss = torch.mean(neg_log_prob * discounted_reward)
        return

    def _discount_and_norm_rewards(self, true_reward):
        # to be done,
        return true_reward

