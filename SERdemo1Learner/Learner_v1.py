import RLbrain_v1
from RLbrain_v1 import Agent

import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import socketserver, socket, selectors, traceback
host = '127.0.0.1'
port = 65432
sel = selectors.DefaultSelector()


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

def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    print("accepted connection from", addr)
    conn.setblocking(False)
    message = socketserver.Message(sel, conn, addr)
    sel.register(conn, selectors.EVENT_READ, data=message)

def socket_init():
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Avoid bind() exception: OSError: [Errno 48] Address already in use
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((host, port))
    lsock.listen()
    print("listening on", (host, port))
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)
    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    message = key.data
                    try:
                        message.process_events(mask)
                    except Exception:
                        print(
                            "main: error: exception for",
                            f"{message.addr}:\n{traceback.format_exc()}",
                        )
                        message.close()
    finally: sel.close()

def send_parameters():
    # socket send parameters back to worker
    # when receive newest parameter request, call this
    return


def receive_exp(learner):
    # socket receive experience, then perform training
    # when receive experience from workers, call this
    RLbrain_v1.training_process(learner)
    return

def training_process(learner):
    criterion = RLbrain_v1.MyLoss()
    transitions = memory.sample(memory.position)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state, dim=0)
    action_batch = torch.stack(batch.action, dim=0)
    reward_batch = torch.stack(batch.reward, dim=0)

    q_pred = learner(state_batch)
    # loss = criterion(q_pred,action_batch, RLbrain_v1.MyLoss.discount_and_norm_rewards(reward_batch))
    loss = criterion(q_pred, action_batch, reward_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_dict.append(loss.item())
    return

lr = 0.01
learner = Agent(num_actions=6)
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(learner.parameters(), lr)
loss_dict = []

if __name__ == "__main__":
    state1 = torch.Tensor(
        [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action1 = torch.LongTensor([5])
    reward1 = torch.Tensor([0])
    state2 = torch.Tensor([1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action2 = torch.LongTensor([1])
    reward2 = torch.Tensor([10])
    state3 = torch.Tensor([1.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, 0.710891563807, 0.00743301723248, 0.0516815336039])
    action3 = torch.LongTensor([0])
    reward3 = torch.Tensor([20])
    state4 = torch.Tensor([2.000090025996463, 2.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         1.9999998758519482, 0.710891563807, 0.00743301723248, 0.000])
    action4 = torch.LongTensor([0])
    reward4 = torch.Tensor([10])
    state5 = torch.Tensor(
        [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action5 = torch.LongTensor([1])
    reward5 = torch.Tensor([10])
    state6 = torch.Tensor(
        [1.000090025996463, 1.115451599181422, 0.9956611980375802, 1.0000346655401584, 1.0000041727872926,
         0.9999998758519482, -0.710891563807, -0.00743301723248, 0.0516815336039])
    action6 = torch.LongTensor([0])
    reward6 = torch.Tensor([20])
    memory.push(state1, action1, reward1)
    memory.push(state2, action2, reward2)
    memory.push(state3, action3, reward3)
    memory.push(state4, action4, reward4)
    memory.push(state5, action5, reward5)
    memory.push(state6, action6, reward6)
    for i in range(100):
        training_process(learner)
    print(loss_dict)

    print(learner.select_action(state1))
    # socket listen


