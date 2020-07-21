import RLbrain_v1
from RLbrain_v1 import Agent

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
host = '172.19.0.1'
port = 65432
sel = selectors.DefaultSelector()


device = torch.device('cuda')

Transition = namedtuple('Transition', ('state', 'action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.core_memory = []
        self.core_position = 0

    def push(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push_core(self, *args):
        if len(self.core_memory) < self.capacity:
            self.core_memory.append(None)
        self.core_memory[self.core_position] = Transition(*args)
        self.core_position = (self.core_position + 1) % self.capacity

    def sample(self, batch_size):
        this_batch_size = min(batch_size, self.position)
        if self.position == 0:
            return []
        return random.sample(self.memory, this_batch_size)
    
    def sample_core(self, batch_size):
        this_batch_size = min(batch_size, self.core_position)
        if self.core_position == 0:
            return []
        return random.sample(self.core_memory, this_batch_size)

    def __len__(self):
        return len(self.memory)


class Learner():

    def __init__(self, lr=0.01, mem_capacity=10000, num_actions=12):
        self.lr = lr
        self.agent = Agent(num_actions=num_actions)
        self.replay_memory = ReplayMemory(mem_capacity)
        self.optimizer = optim.RMSprop(self.agent.parameters(), self.lr)
        self.loss_dict = []
        self.state_dict = self.agent.state_dict()

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
        for exp in encoded_data:
            self.replay_memory.push(exp['state'], exp['action'], exp['reward'])
            if exp['reward'] >= 10:
                self.replay_memory.push_core(exp['state'], exp['action'], exp['reward'])
        print ("Got new stuff with len: ", len(data))
        # # TODO: CORE memory, for games with large reward.
        for i in range(3):
            self.training_process()
        self.state_dict = self.agent.state_dict()
        torch.save(self.agent.state_dict(), 'params.pkl')
        print(self.loss_dict)

    def training_process(self):
        criterion = RLbrain_v1.MyLoss()
        # @@@@@ batch size of each update?
        #print('1',self.replay_memory.sample(64))
        #print('2',self.replay_memory.sample_core(self.replay_memory.core_position))
        transitions = self.replay_memory.sample(128) + self.replay_memory.sample_core(self.replay_memory.core_position)
        #print('3', transitions)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state, dim=0)
        action_batch = torch.stack(batch.action, dim=0)
        reward_batch = torch.stack(batch.reward, dim=0)

        q_pred = self.agent(state_batch)
        # loss = criterion(q_pred,action_batch, RLbrain_v1.MyLoss.discount_and_norm_rewards(reward_batch))
        loss = criterion(q_pred, action_batch, reward_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_dict.append(loss.item())
        return


# Do the main work here?
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
    # action1 = torch.LongTensor([10])
    # reward1 = torch.Tensor([0])
    # state2 = torch.Tensor([1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      1.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action2 = torch.LongTensor([10])
    # reward2 = torch.Tensor([10])
    # state3 = torch.Tensor([1.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      1.9999998758519482, 0.710891563807, 0.00743301723248, 0.0516815336039])
    # action3 = torch.LongTensor([10])
    # reward3 = torch.Tensor([20])
    # state4 = torch.Tensor([2.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      1.9999998758519482, 0.710891563807, 0.00743301723248, 0.000])
    # action4 = torch.LongTensor([0])
    # reward4 = torch.Tensor([10])
    # state5 = torch.Tensor(
    #     [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action5 = torch.LongTensor([9])
    # reward5 = torch.Tensor([10])
    # state6 = torch.Tensor(
    #     [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
    #      0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    # action6 = torch.LongTensor([9])
    # reward6 = torch.Tensor([20])
    # memory.push(state1, action1, reward1)
    # memory.push(state2, action2, reward2)
    # memory.push(state3, action3, reward3)
    # memory.push(state4, action4, reward4)
    # memory.push(state5, action5, reward5)
    # memory.push(state6, action6, reward6)
    # for i in range(2):
    #     training_process(learner)
    # print(loss_dict)

    # print(learner.select_action(state1))
    # socket listen
