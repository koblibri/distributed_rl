import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, num_actions=12, input_size=9, fc_size=512, lstm_hidden=256, isactor=False):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.input_size = input_size
        self.fc_size = fc_size
        self.lstm_hidden = lstm_hidden
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, fc_size - 2)

        self.lstm = nn.LSTMCell(fc_size, lstm_hidden)
        # batch_first=True: the input and output tensors are provided as (batch, seq, feature)

        self.head = Head(num_actions)
        self.isactor = isactor

    def forward(self, x, action, reward, dones=None, core_state=None, batch_size=1, isactor=False):
        # seq_len, bs, x, last_action, reward = combine_time_batch(x, last_action, reward, actor)
        # x = torch.flatten(x)
        # print(x.shape)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        # isactor = self.isactor

        if isactor:
            seq_len = 1
            batch_size = 1
        else:
            seq_len = x.shape[0]
            batch_size = x.shape[1]

        x = x.view(seq_len, batch_size, -1)
        action = action.view(seq_len, batch_size, -1)
        reward = reward.view(seq_len, batch_size, -1)
        # print('x.shape', x.shape)
        # print('action.shape', action.shape)
        # print('reward.shape', reward.shape)
        x = torch.cat((x, action, reward), dim=2)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        # print(x.tolist()[0:15])
        # x = x.view()
        # print(x.shape)
        hx = None
        cx = None

        if core_state is None:
            hx = torch.randn((batch_size, self.lstm_hidden))
            cx = torch.randn((batch_size, self.lstm_hidden))
        else:
            hx = core_state[0]
            cx = core_state[1]

        # print(hx.shape)
        lstm_out = []
        # print(seq_len)
        for i in range(seq_len):
            hx, cx = self.lstm(x[i], (hx, cx))
            # lstm_out = torch.cat((lstm_out, hx), dim=0)
            lstm_out.append(hx)
            core_state = torch.stack([hx, cx], 0)

        # for state, done in zip(torch.unbind(x, 0), torch)
        # print('lstm_out.len', lstm_out.shape)
        x = torch.cat(lstm_out, 0)
        # print(x.shape)
        new_action, policy_logits, baseline = self.head(x)

        if isactor:
            return new_action, policy_logits.view(1, -1), core_state
        else:
            # print(policy_logits.view(seq_len, -1, batch_size).shape)
            return policy_logits.view(seq_len, -1, batch_size), baseline.view(seq_len, batch_size)


class Head(nn.Module):
    def __init__(self, num_actions):
        super(Head, self).__init__()
        self.actor_linear = nn.Linear(256, num_actions)
        self.critic_linear = nn.Linear(256, 1)

    def forward(self, x):
        policy_logits = self.actor_linear(x)
        baseline = self.critic_linear(x)
        prob_weights = F.softmax(policy_logits, dim=0).clamp(1e-10, 1)
        # print(prob_weights)

        new_action = torch.multinomial(prob_weights, 1, replacement=True)
        return new_action, policy_logits, baseline


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.num_actions = 12

    # def forward(self, q_pred, true_action, discounted_reward):
    #     # define the loss, old version, only policy gradient
    #     one_hot = torch.zeros(
    #         len(true_action), self.num_actions).scatter_(1, true_action, 1)
    #     neg_log_prob = torch.sum(-torch.log(F.softmax(q_pred, dim=1)) * one_hot, dim=1)
    #     loss = torch.mean(neg_log_prob * discounted_reward)
    #     return loss

    # def discount_and_norm_rewards(self, true_reward):
    #     # to be done
    #     return true_reward

    def compute_baseline_loss(self, advantages):
        # Loss for the baseline, summed over the time dimension.
        # Multiply by 0.5 to match the standard update rule:
        # d(loss) / d(baseline) = advantage
        return 0.5 * torch.mean(advantages**2)

    def compute_entropy_loss(self, logits):
        policy = F.softmax(logits, dim=1)
        log_policy = F.log_softmax(logits, dim=1)
        entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
        return -torch.mean(entropy_per_timestep)

    def compute_policy_gradient_loss(self, logits, actions, advantages):
        cross_entropy = F.cross_entropy(logits, actions)
        # print(advantages.requires_grad)
        # advantages = advantages.no_grad()
        policy_gradient_loss_per_timestep = cross_entropy * advantages
        return torch.mean(policy_gradient_loss_per_timestep)


if __name__ == "__main__":
    state1 = torch.Tensor(
        [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action1 = torch.Tensor([10])
    reward1 = torch.Tensor([5])
    state2 = torch.Tensor([1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action2 = torch.Tensor([5])
    reward2 = torch.Tensor([20])
    state3 = torch.Tensor([1.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, 0.710891563807, 0.00743301723248, 0.0516815336039])
    action3 = torch.Tensor([6])
    reward3 = torch.Tensor([30])
    state4 = torch.Tensor([2.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, 0.710891563807, 0.00743301723248, 0.000])
    action4 = torch.Tensor([7])
    print(action4.shape)
    reward4 = torch.Tensor([40])
    state5 = torch.Tensor(
        [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action5 = torch.Tensor([9])
    reward5 = torch.Tensor([50])
    state6 = torch.Tensor(
        [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action6 = torch.Tensor([12])
    reward6 = torch.Tensor([60])
    x = torch.stack([state1, state2, state3, state4, state5, state6], 0)
    # x = x.view(x.shape[0], 1, x.shape[1])
    # print(x)
    x = torch.stack([x, x], 0)
    # x = x.permute(1, 0, 2)
    a = [action1, action2, action3, action4, action5, action6]
    action = torch.stack(a, 0)
    action = torch.stack([action, action], 0)
    reward = torch.stack([reward1, reward2, reward3, reward4, reward5, reward6], 0)
    reward = torch.stack([reward, reward], 0)

    # print('x.shape', x.shape)
    # print('action.shape', action.shape)
    # print('reward.shape', reward.shape)
    x = x.permute(1, 0, 2)
    action = torch.transpose(action, 0, 1)
    reward = torch.transpose(reward, 0, 1)
    # print('x.shape', x.shape)
    # print('action.shape', action.shape)
    # print('reward.shape', reward.shape)

    test_agent = Agent()
    result = test_agent(x, action, reward, isactor=False)
    print(result)
