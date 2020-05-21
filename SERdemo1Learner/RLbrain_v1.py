import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.l1 = nn.Linear(7, 128)  # 7 input state, how many? three, only x y z??
        self.l2 = nn.Linear(128, num_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.flatten(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        y = F.softmax(x)  # output the action-value(Qvalue) of each action  || maybe not need this softmax, just logit
        return y

    def select_action(self, state):
        prob_weights = self.forward(state)
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)  # @@@ maybe need transfer to list here
        return action



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
        # to be done
        return true_reward


