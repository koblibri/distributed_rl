import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.l1 = nn.Linear(9, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, num_actions)

    def forward(self, x):
        # x = torch.flatten(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        y = self.l4(x)
        # y = F.softmax(x)  # output the action-value(Qvalue) of each action  || maybe not need this softmax, just logit
        return y

    def select_action(self, state):
        prob_weights = F.softmax(self.forward(state), dim=0).clamp(1e-10, 1)
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
        one_hot = torch.zeros(
            len(true_action), self.num_actions).scatter_(1, true_action, 1)
        # print('true_action:', true_action)
        # print('q_pred:', q_pred)
        # print('one_hot:', one_hot)
        neg_log_prob = torch.sum(-torch.log(F.softmax(q_pred,
                                                      dim=1)) * one_hot, dim=1)
        # print('neg_log_prob:', neg_log_prob)
        loss = torch.mean(neg_log_prob * discounted_reward)
        # print('loss :', loss )
        return loss

    def discount_and_norm_rewards(self, true_reward):
        # to be done
        return true_reward
