import traceback
import socket
import socketclient
import time
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

import random

# if gpu is to be used
device = torch.device('cuda')

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

try:
    import selectors
except ImportError:
    import selectors2 as selectors  # run  python -m pip install selectors2
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
                        "main: error: exception for %s \n %s"
                        % (message.addr, traceback.format_exc())
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


def send_exp():  # socket send experience to learner
    # TODO: Enter new Experiences here
    exp = memory.sample(memory.position)
    request = create_request("push", exp)  # here, experiences are Tensor, may cause bugs
    start_connection(host, port, request)

    wait_response()
    # TODO: clear memory
    return


def compute_reward(old_state, new_state, init_state):
    """computes the reward for an actio nold_state -> new_state

    :param old_state: robot state before action
    :type old_state: ( _ ,geometry_msgs.msg._Pose.Pose)
    :param new_state: robot state after action
    :type new_state: ( _ ,geometry_msgs.msg._Pose.Pose)
    :param init_state: initial robot state
    :type init_state: ( _ ,geometry_msgs.msg._Pose.Pose)
    :return: reward for this action
    :rtype: int
    """
    # compare state, if position changed get reward. if z axis is lower than initial, get big reward.
    x_new = new_state[1].position.x
    y_new = new_state[1].position.y
    z_new = new_state[1].position.z

    new_position_np = np.array((x_new, y_new, z_new))

    x_old = old_state[1].position.x
    y_old = old_state[1].position.y
    z_old = old_state[1].position.z

    old_position_np = np.array((x_old, y_old, z_old))

    distance = np.linalg.norm(new_position_np - old_position_np)

    z_init = init_state[1].position.z
    # check if blue object fell from the table
    eps = 0.1
    if z_new + eps < z_init:
        return 100

    # check if blue object was moved
    if(distance > eps):
        return 10

    return 1


def select_strategy(strategy_threshold):
    """ select strategy (explore or exploit) for a given threshold 

    :param strategy_threshold: probability threshold
    :type strategy_threshold: int
    :return: strategy (explore or exploit) for the next round
    :rtype: String
    """
    prob = random.uniform(0, 1)
    strategy = 'exploit'
    if prob < strategy_threshold:
        strategy = 'explore'
    return strategy


def check_done(new_state, init_state):
    """check, if robot is done (blue object fell from the table)

    :param new_state: robot state after action
    :type new_state:  ( _ ,geometry_msgs.msg._Pose.Pose)
    :param init_state: initial robot state
    :type init_state: ( _ ,geometry_msgs.msg._Pose.Pose)
    :return: true, if object has fallen, else false
    :rtype: bool
    """
    # + 0.1 because there are some noises
    if((new_state[1].position.z + 0.1) < init_state[1].position.z):
        return True
    else:
        return False


def transfer_action(current_joint_state, action):
    """ transfer action from {0-11} to {-6,...,-1, 1,...,6},
        where +6: joint6 +1, -6: joint6 -1,
                +5: joint5 +1, -5: joint5 -1,
                +4: joint4 +1, -4: joint4 -1,
                +3: joint3 +1, -3: joint3 -1,
                +2: joint2 +1, -2: joint2 -1,
                +1: joint1 +1, -1: joint1 -1,
    :param current_joint_state: robot current joint values
            type: list, e.g.: [1,-2,0,0,0,0]
    :param action: selected action of this step
            type: int,  e.g.: 9 or 4
    :return: new_joint_state, the new robot joint values, after transferred to j1,j2,j3..., can be used in robot.act()
    :rtype: list
    """
    action -= 5
    if action <= 0:
        action -= 1
    act_joint_num = torch.abs(action) - 1
    new_joint_state = current_joint_state[act_joint_num] + torch.sign(action)
    return new_joint_state


robot = experiment_api.Robot()
time.sleep(1)
memory = ReplayMemory(10000)
worker = Agent(num_actions=12)  # now num_action is 12, because each joint has two direction!
                                #  actions transferred by transfer_action(current_joint_state, action)

num_episodes = 50  # 50 rounds of games
for i in range(num_episodes):
    robot.reset()
    init_state = robot.get_current_state()  # @@@@ here using self.object_init_state!!
    state = init_state
    strategy_threshold = 1 - 1/(num_episodes - i)
    strategy = select_strategy(strategy_threshold)
    for actions_counter in count():

        # object state has to be transferred at here,
        # because otherwise in Learner, we cannot parse state by state.position
        list_state = list(state[0]) + [state[1].position.x, state[1].position.y, state[1].position.z]
        tensor_state = torch.Tensor(list_state)
        action = None
        new_joint_state = []
        if strategy == 'exploit':
            # action = worker(state)
            action = worker.select_action(tensor_state)
            current_joint_state = list(state[0])
            new_joint_state = transfer_action(current_joint_state, action)
        else:
            action = random.uniform(0, 11)
            current_joint_state = list(state[0])
            new_joint_state = transfer_action(current_joint_state, action)

            # for i in range(6):
            #     action.append(random.uniform(-3, 3))
        j1, j2, j3, j4, j5, j6 = new_joint_state
        robot.act(j1, j2, j3, j4, j5, j6)

        time.sleep(3)  # sleep wait for this action finished  @@@ to be done: speed up the robot joint!!!
        new_state = robot.get_current_state()
        reward = compute_reward(state, new_state, init_state)

        tensor_action = torch.LongTensor([action])
        tensor_reward = torch.Tensor([reward])
        memory.push(tensor_state, tensor_action, tensor_reward)

        state = new_state

        if check_done(new_state, init_state):
            break

    send_exp()  # one game over, send the experience
    print('number of actions in this round game:', actions_counter)

print(num_episodes, ' training over')
