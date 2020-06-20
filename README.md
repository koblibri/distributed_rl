# Distributed Reinforcement Learning Agent

## Installing packages with pip

```bash
pip install -r requirements.txt --no-cache-dir
```

## Socket Inits:
    Configure your IPs! 
    The IP in Learner_v1 is the Server-IP and must be the same as IP in Worker_v1. 
    Either work locally inside the Docker-Container and use localhost 127.0.0.1
    Or have Worker inside the container and attach the Learner to the Docker-Host. Use 'docker network inspect nrpnet' to find out the IP of your Docker-Host. Mine was 172.19.0.1

This repository contains files for a Reinforcement Learning Experiment on the Neurorobotics platform
(NRP).

## Import experiment files
To load the experiment in your installation of the NRP (either local or source install), open the
NRP frontend, and then navigate to the 'My experiments' tab. There, click on the 'Import folder'
button, and select the experiment/ser_rl_ss20 folder to upload. After that, the experiment should
appear under 'My experiments', and you should be able to launch it.

## Experiment Setup
The environment consists of one Hollie Robot arm with six Degrees of Freedom, sitting on a table. A
Schunk hand is mounted on the arm, but it is of little relevance to the task to be solved. There is
also one blue cylinder on the table.

<img src="experiment/ser_rl_ss20/ExDDemoManipulation.png" width="400" />

The task is for the arm to move towards the cylinder and knock it off the table. The observations at
each time step are: 
* The current joint positions (six dimensional)
* The current object pose (position and orientation) (seven dimensional)

and the actions to be taken are:
* The six joint positions

A position controller takes care of moving the joints to the desired positions.

## Interacting with the Simulation
After launching the experiment and clicking on the play button, you can interact with the simulation
from a python shell though the Agent class in 'experiment_api.py'. It is better to do this within
the docker container, because you might need to install additional dependencies if you want to run 
it on your system. Below are the steps for interacting with the simulation from within the docker 
container:

1. Copy the experiment_api.py file to the backend container:
```
$ docker cp experiment_api.py nrp:/home/bbpnrsoa/
```

2. Access the backend docker container:
```
$ docker exec -it nrp bash
```

3. Open a python shell inside the backend container and import the experiment api:
```
$ cd ~

$ python

>>> import experiment_api
```

4. Instantiate the agent class and explore the available functions:
```
>>> agent = experiment_api.Agent()
>>> agent.get_current_state()
>>> agent.act(1, 1, 1, 1, 1, 1)
>>> agent.reset()
```

Feel free to extend the experiment_api.py with functions that you see necessary.
