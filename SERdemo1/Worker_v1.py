import experiment_api
import RLbrain_v1
from RLbrain_v1 import Agent

import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# if gpu is to be used
device = torch.device('cuda')

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

import socket, selectors
import SERdemo1.socketclient as socketclient
sel = selectors.DefaultSelector()
host = '127.0.0.1'
port = 65432


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

def create_request(action, value):
    if action == "pull":
        return dict(
            type="binary/pull",
            encoding="binary",
            content=None,
        )
    else:
        return dict(
            type="binary/push",
            encoding="binary",
            content=bytes(value),
        )


def start_connection(host, port, request):
    addr = (host, port)
    print("starting connection to", addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = socketclient.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)

def wait_response():
    try:
        while True:
            events = sel.select(timeout=1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask)
                except Exception:
                    print(
                        "main: error: exception for",
                        f"{message.addr}:\n{traceback.format_exc()}",
                    )
                    message.close()
            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    finally:
        sel.close()

def pull_parameters():
    request = create_request("pull", None)
    start_connection(host, port, request)
    # socket get learner side .state_dict()
    wait_response()

def send_exp():
    request = create_request("push", [1,2,3] ) #TODO: Enter new Experiences here
    start_connection(host, port, request)
    # socket send experience to learner
    wait_response()
    return


def compute_reward(old_state, new_state, init_state):
    # compare state, if position changed get reward. if z axis is lower than initial, get big reward.
    return 1


def check_done(new_state, init_state):
    # check the game is over or not, @@@@@add a timer here!
    return True


robot = experiment_api.Robot()
worker = Agent(num_actions=6)


memory = ReplayMemory(10000)


num_episodes = 50
for i in range(num_episodes):
    robot.reset()
    init_state = robot.get_current_state()
    state = init_state
    worker.load_state_dict(pull_parameters()) # before every game, request newest parameters

    for actions_counter in count():
        action = worker.select_action(state)
        j1, j2, j3, j4, j5, j6 = action
        robot.act(j1, j2, j3, j4, j5, j6)
        new_state = robot.get_current_state()
        reward = compute_reward(state, new_state, init_state)

        memory.push(state, action, reward)

        state = new_state

        if check_done():
            send_exp()  # one game over, send the experience
            print('number of actions in this round game:', actions_counter)
            break

print(num_episodes, ' training over')



