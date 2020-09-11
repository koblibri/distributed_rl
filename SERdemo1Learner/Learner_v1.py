import RLbrain_v1
from RLbrain_v1 import Agent

import vtrace

import pickle
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from socketserver import Message as sockMessage
import socket
import traceback
import threading
import time
try:
    import selectors
except ImportError:
    import selectors2 as selectors  # run  python -m pip install selectors2
host = '127.0.0.1'
port = 65432
sel = selectors.DefaultSelector()


device = torch.device('cuda')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'logits'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        # self.core_memory = []
        # self.core_position = 0

    def push(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size=128):
        # now, trajectories are not too much, so sample all, batch_size is not used
        # this_batch_size = min(batch_size, self.position)
        if self.position == 0:
            print('error: empty memory when sampling')
            return []
        # return random.sample(self.memory, this_batch_size)
        trajectory = [self.memory[0]]
        self.memory.pop(0)
        self.position -= 1
        return trajectory

    def __len__(self):
        return len(self.memory)


class Learner():

    def __init__(self, lr=0.001, mem_capacity=10000, num_actions=12):
        self.lr = lr
        self.agent = Agent(num_actions=num_actions)
        self.replay_memory = ReplayMemory(mem_capacity)
        self.optimizer = optim.RMSprop(self.agent.parameters(), self.lr)
        self.loss_dict = []
        self.state_dict = self.agent.state_dict()

        self.gamma = 0.99 # 'Discounting factor.'
        self.baseline_cost = 0.5
        self.entropy_cost = 0.00025

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()  # Should be ready to read
        print("accepted connection from", addr)
        conn.setblocking(True)
        message = sockMessage(sel, conn, addr, self.send_parameters())
        sel.register(conn, selectors.EVENT_READ, data=message)

    def socket_init(self):
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Avoid bind() exception: OSError: [Errno 48] Address already in use
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((host, port))
        lsock.listen(5)
        print("listening on", (host, port))
        lsock.setblocking(False)
        sel.register(lsock, selectors.EVENT_READ, data=None)
        try:
            while True:
                events = sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        self.accept_wrapper(key.fileobj)
                    else:
                        message = key.data
                        try:
                            exp = message.process_events(mask)
                            if exp != None:
                                self.receive_exp(exp)
                        except Exception:
                            print(
                                "main: error: exception for %s\n %s"
                                % (message.addr, traceback.format_exc))
                            traceback.print_exc()
                            message.close()
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
        finally:
            sel.close()

    def send_parameters(self):
        # socket send parameters back to worker
        # when receive newest parameter request, call this
        # return pickle.dumps(self.agent.state_dict())
        return pickle.dumps(self.state_dict)

    def receive_exp(self, data):
        encoded_data = pickle.loads(data)
        # if float(encoded_data[-1]['reward']) >= 19:
        state_list = []
        action_list = []
        reward_list = []
        done_list = []
        logits_list = []
        for exp in encoded_data:
            state_list.append(exp['state'])
            action_list.append(exp['action'])
            reward_list.append(exp['reward'])
            done_list.append(exp['done'])
            logits_list.append(exp['logits'])
            # self.replay_memory.push(exp['state'], exp['action'], exp['reward'])
            # if exp['reward'] >= 10:
            #     self.replay_memory.push_core(exp['state'], exp['action'], exp['reward'])
        state_trajectory = torch.stack(state_list, 0)
        action_trajectory = torch.stack(action_list, 0)
        reward_trajectory = torch.stack(reward_list, 0)
        done_trajectory = torch.stack(done_list, 0)
        logits_trajectory = torch.stack(logits_list, 0)

        self.replay_memory.push(state_trajectory, action_trajectory, reward_trajectory, done_trajectory, logits_trajectory)

        print ("Got new stuff with len: ", len(data))
        # # TODO: CORE memory, for games with large reward.
        # for i in range(3):
        #     self.training_process()
        while self.replay_memory.position != 0:
        # for i in range(5):
            self.training_process()
        self.state_dict = self.agent.state_dict()
        torch.save(self.agent.state_dict(), 'params.pkl')
        print(self.loss_dict)

    def training_process(self):
        # criterion = RLbrain_v1.MyLoss()
        # @@@@@ batch size of each update?
        #print('1',self.replay_memory.sample(64))
        #print('2',self.replay_memory.sample_core(self.replay_memory.core_position))
        # transitions = self.replay_memory.sample(128) + self.replay_memory.sample_core(self.replay_memory.core_position)
        #print('3', transitions)
        transitions = self.replay_memory.sample()
        batch = Transition(*zip(*transitions))
        # print(batch.reward[0])
        state_batch = torch.stack(batch.state, dim=0)
        action_batch = torch.stack(batch.action, dim=0)
        reward_batch = torch.stack(batch.reward, dim=0)
        done_batch = torch.stack(batch.done, dim=0)
        behavior_logits_batch = torch.stack(batch.logits, dim=0)

        # print('state_batch.shape', state_batch.shape)
        # print('action_batch.shape', action_batch.shape)
        # print('reward_batch.shape', reward_batch.shape)
        # print('done_batch.shape', done_batch.shape)
        # print('behavior_logits_batch.shape', behavior_logits_batch.shape)

        # make time major
        state_batch = torch.transpose(state_batch, 0, 1)
        action_batch = torch.transpose(action_batch, 0, 1)
        reward_batch = torch.transpose(reward_batch, 0, 1)
        done_batch = torch.transpose(done_batch, 0, 1)
        if len(behavior_logits_batch.shape) == 4:
            behavior_logits_batch = behavior_logits_batch.squeeze(2)
        behavior_logits_batch = behavior_logits_batch.permute(1, 2, 0)

        # print('state_batch.shape', state_batch.shape)
        # print('action_batch.shape', action_batch.shape)
        # print('reward_batch.shape', reward_batch.shape)
        # print('done_batch.shape', done_batch.shape)
        # print('behavior_logits_batch.shape', behavior_logits_batch.shape)

        target_logits, baseline = self.agent(x=state_batch, action=action_batch, reward=reward_batch, dones=done_batch, core_state=None, isactor=False)
        # print(target_logits)
        # print('------------------')
        # print(baseline)
        # print('target_logits', target_logits)
        # print('behavior_logits_batch', behavior_logits_batch)
        # print(target_logits.shape)
        # print(baseline.shape)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]

        actions, behaviour_logits, rewards, dones = action_batch.view(action_batch.shape[0], -1).type(torch.long)[1:], behavior_logits_batch[1:], \
                                                    reward_batch.view(reward_batch.shape[0], -1)[1:], done_batch.view(done_batch.shape[0], -1)[1:]

        target_logits, baseline = target_logits[:-1], baseline[:-1]
        # target_logits, baseline = target_logits[1:], baseline[1:]
        # print(actions.shape)
        # actions, behaviour_logits, rewards, dones = action_batch.view(action_batch.shape[0], -1).type(torch.long), behavior_logits_batch, \
        #                                             reward_batch.view(reward_batch.shape[0], -1), done_batch
        # target_logits, baseline = target_logits, baseline

        discounts = (~dones).float() * self.gamma

        vs, pg_advantages = vtrace.from_logits(
            behaviour_policy_logits=behaviour_logits,
            target_policy_logits=target_logits,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value)
        # print('vs', vs)
        # print('pg_advantages', pg_advantages)
        # print('baseline', baseline)
        self.optimizer.zero_grad()
        criterion = RLbrain_v1.MyLoss()

        loss = criterion.compute_policy_gradient_loss(target_logits, actions, pg_advantages)
        loss += self.baseline_cost * criterion.compute_baseline_loss(vs - baseline)

        loss += self.entropy_cost * criterion.compute_entropy_loss(target_logits)
        # print(loss)

        loss.backward()

        self.optimizer.step()

        self.loss_dict.append(loss.item())
        return


# Does the main work here?
def learner_func():
    while True:
        time.sleep(1)


x = threading.Thread(target=learner_func, args=())
# x.start()

learner = Learner()
if os.path.exists('params.pkl'):
    learner.agent.load_state_dict(torch.load('params.pkl'))

if __name__ == "__main__":
    # x.join()
    learner.socket_init()

    # state1 = torch.Tensor(
    #     [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action1 = torch.Tensor([0])
    # reward1 = torch.Tensor([0.1])
    # state2 = torch.Tensor(
    #     [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      1.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action2 = torch.Tensor([1])
    # reward2 = torch.Tensor([0.2])
    # state3 = torch.Tensor(
    #     [1.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      1.9999998758519482, 0.710891563807, 0.00743301723248, 0.0516815336039])
    # action3 = torch.Tensor([2])
    # reward3 = torch.Tensor([0.3])
    # state4 = torch.Tensor(
    #     [2.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      1.9999998758519482, 0.710891563807, 0.00743301723248, 0.000])
    # action4 = torch.Tensor([3])
    # reward4 = torch.Tensor([0.4])
    # state5 = torch.Tensor(
    #     [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action5 = torch.Tensor([4])
    # reward5 = torch.Tensor([0.5])
    # state6 = torch.Tensor(
    #     [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action6 = torch.Tensor([5])
    # reward6 = torch.Tensor([0.6])
    # x = torch.stack([state1, state2, state3, state4, state5, state6], 0)
    # # x = x.view(x.shape[0], 1, x.shape[1])
    # # print(x)
    # # x = torch.stack([x], 0)
    # # x = x.permute(1, 0, 2)
    # # a = [action1, action2, action3, action4, action5, action6]
    # action = torch.stack([action1, action2, action3, action4, action5, action6], 0)
    # # print(action.shape)
    # # action = torch.stack([action], 0)
    # reward = torch.stack([reward1, reward2, reward3, reward4, reward5, reward6], 0)
    # # reward = torch.stack([reward], 0)
    # done1 = torch.tensor(True, dtype=torch.bool)
    # done0 = torch.tensor(False, dtype=torch.bool)
    # dones = torch.stack([done0, done0, done0, done0, done0, done1], 0)
    #
    # logit1 = torch.Tensor([-0.3957,  0.2553,  0.1731,  0.1928, -0.1930,  0.2238, -0.1347, -0.0121,
    #       0.3904, -0.0646,  0.0054, -0.0808])
    # logit2 = torch.Tensor([-0.1287,  0.0996,  0.0541,  0.1162, -0.0909,  0.1186, -0.0578,  0.0170,
    #       0.2253, -0.0636, -0.1721, -0.0074])
    # logit3 = torch.Tensor([ 0.0329, -0.0275, -0.0688, -0.0430,  0.1320,  0.0422, -0.1675, -0.1041,
    #       0.0313, -0.1199, -0.0995,  0.1064])
    # logit4 = torch.Tensor([ 0.0769,  0.0710,  0.0164,  0.0278, -0.0437,  0.0261, -0.0264, -0.0075,
    #       0.1767, -0.0654, -0.2793,  0.0286])
    # logit5 = torch.Tensor([ 0.1634,  0.0215, -0.0679, -0.0599,  0.0825, -0.0294, -0.0768, -0.1279,
    #       0.0404, -0.1553, -0.2277,  0.1225])
    # logit6 = torch.Tensor([-0.0850, -0.2342, -0.2102,  0.1763,  0.4641, -0.1137,  0.1349,  0.2146,
    #      -0.2993, -0.2112, -0.0011, -0.1569])
    #
    # logits = torch.stack([logit1, logit2, logit3, logit4, logit5, logit6], 0)
    # # print('logits.shape', logits.shape)
    # x = torch.stack([state1, state2], 0)
    # action = torch.stack([action1, action2], 0)
    # reward = torch.stack([reward1, reward2], 0)
    # dones = torch.stack([done0, done1], 0)
    # logits = torch.stack([logit1, logit2], 0)
    #
    # learner.replay_memory.push(x, action, reward, dones, logits)
    # learner.replay_memory.push(x, action, reward, dones, logits)
    # learner.replay_memory.push(x, action, reward, dones, logits)
    # learner.replay_memory.push(x, action, reward, dones, logits)
    # learner.replay_memory.push(x, action, reward, dones, logits)
    # #
    # while learner.replay_memory.position != 0:
    #     learner.training_process()
    # print(learner.loss_dict)
    #
    #
    #
    #
    # # actor testing:
    # # state1 = state1.view(1, 1, 9)
    # # action1 = action1.view(1, 1, 1)
    # # reward1 = reward1.view(1, 1, 1)
    # # na, pl, _ = learner.agent(state1, action1, reward1, None, isactor=True)
    # state = torch.stack([state1, state2], 0)
    # action = torch.stack([action1, action2], 0)
    # reward = torch.stack([reward1, reward2], 0)
    # dones = torch.stack([done0, done1], 0)

    # na, pl, _ = learner.agent(state, action, reward, dones, isactor=False)
    # print(na)
    # print(pl)


