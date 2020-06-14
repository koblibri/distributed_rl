import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        # 7 input state, how many? three, only x y z??
        self.l1 = nn.Linear(9, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, num_actions)

    def forward(self, x):
        """forward pass for a robot state

        :param x: tuple of position of blue object and pose of the robot
        :type x: ((float, float, float),geometry_msgs.msg._Pose.Pose)
        :return: actions for the next step
        :rtype: List[float]
        """
        # change tuple to array, add position of object
        # parameters = list(x[0]) + [x[1].position.x,
        #                            x[1].position.y, x[1].position.z]
        # x = torch.tensor(parameters)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        y = self.l4(x)
        # linear activation no extra function necessary
        return y

    def select_action(self, x):
        # print(x)
        # parameters = list(x[0]) + [x[1].position.x, x[1].position.y, x[1].position.z]
        prob_weights = F.softmax(self.forward(x), dim=0).clamp(1e-10, 1)
        print(prob_weights)

        # action = np.random.sample(range(prob_weights.shape[0]), p=prob_weights) # @@@ maybe need transfer to list here
        # action = np.random.choice(prob_weights.shape[0], 1, p=prob_weights, replace=False)
        action = torch.multinomial(prob_weights, 1, replacement=False)
        return action


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.num_actions = 12

    def forward(self, q_pred, true_action, discounted_reward):
        # define the loss,
        one_hot = torch.zeros(len(true_action), self.num_actions).scatter_(1, true_action, 1)
        # print('true_action:', true_action)
        # print('q_pred:', q_pred)
        # print('one_hot:', one_hot)
        neg_log_prob = torch.sum(-torch.log(F.softmax(q_pred, dim=1)) * one_hot, dim=1)
        # print('neg_log_prob:', neg_log_prob)
        loss = torch.mean(neg_log_prob * discounted_reward)
        # print('loss :', loss )
        return loss

    def discount_and_norm_rewards(self, true_reward):
        # to be done
        return true_reward
