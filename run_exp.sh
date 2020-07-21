#!/bin/bash

cd ~
git clone --progress https://github.com/koblibri/distributed_rl_worker.git
cd distributed_rl_worker
pip install -r requirements.txt --no-cache-dir
#sleep 1
#python cle-virtual-coach start_experiment.py
#sleep 5
#python ./Worker_v1.py
