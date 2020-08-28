# Distributed Reinforcement Learning Agent

## Installing packages with pip

After initializing your backend container, enter the container, copy the repository and install the required packages:

```bash
docker cp . nrp:/home/bbpnrsoa/distributed-reinforcement-learning 
docker exec -it nrp bash
cd /home/bbpnrsoa/distributed-reinforcement-learning
pip install -r requirements.txt --no-cache-dir
```
**Note**: If you run the learner locally, you also have to install the packages in your local environment. (Check if you run Python 2.7. on your local machine, not Python3!) 

## Socket Inits:
Configure your IPs! 
The IP in `SERdemo1Learner/Learner_v1` is the Server-IP and must be the same as IP in `SERdemo1/Worker_v1`. 

Either work inside the Docker-Container and use localhost 127.0.0.1
Or have Worker inside the container and attach the Learner to the Docker-Host. Use 
```bash
docker network inspect nrpnet
```
 to find out the IP of your Docker-Host (e.g 172.19.0.1).


## Import experiment files
To load the experiment in your installation of the NRP (either local or source install), open the
NRP frontend in your browser at location http://host.docker.internal:9000/#/esv-private, and then navigate to the 'My experiments' tab. There, click on the 'Import folder'
button, and select the `SERdemo1/experiment/ser_rl_ss20` folder to upload. After that, the experiment should
appear under 'My experiments', and you should be able to launch it.


## Start Reinforcement Learning:
1. Start Learner either in the container:
```bash
docker exec -it nrp bash
cd /home/bbpnrsoa/distributed-reinforcement-learning
python SERdemo1Learner/Learner_v1.py
```
or start the learner locally:
```bash
python SERdemo1Learner/Learner_v1.py
```
You should see
```bash
('listening on', ('your IP address', 65432))
```
in your console now.

2. Launch the experiment in the NRP frontend. Once the simulation has started, click the play button at the top left to start the robot.

3. Start another bash window, access the container and start the Worker in the container:
```bash
docker exec -it nrp bash
cd /home/bbpnrsoa/distributed-reinforcement-learning
python SERdemo1/Worker_v1.py
```
After the robot inital commands you should see
```bash
('starting connection to', ('your IP address', 65432))
```
in your worker terminal and 
```bash
('accepted connection from', ('your IP address', 34928))
```
in your learner terminal.
## Experiment Setup
The environment consists of one Hollie Robot arm with six Degrees of Freedom, sitting on a table. A
Schunk hand is mounted on the arm, but it is of little relevance to the task to be solved. There is
also one blue cylinder on the table.

<img src="SERdemo1/experiment/ser_rl_ss20/ExDDemoManipulation.png" width="400" />

The task is for the arm to move towards the cylinder and knock it off the table. The observations at
each time step are: 
* The current joint positions (six dimensional)
* The current object pose (position and orientation) (seven dimensional)

and the actions to be taken are:
* The six joint positions

A position controller takes care of moving the joints to the desired positions.

# Besides reinforcement learning 

## Interacting with the Simulation
After launching the experiment and clicking on the play button, you can interact with the simulation
from a python shell though the Agent class in 'experiment_api.py'. It is better to do this within
the docker container, because you might need to install additional dependencies if you want to run 
it on your system. Below are the steps for interacting with the simulation from within the docker 
container:

0. Move to the experiment folder
```
$ cd SERdemo1/experiment/ser_rl_ss20
```

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
