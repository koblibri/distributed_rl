FROM ubuntu:18.04

RUN apt update && apt install -y python3-pip
COPY requirements.txt /home/ubuntu/rl_learner/
WORKDIR /home/ubuntu/rl_learner/
RUN pip3 install -r requirements.txt

COPY __init__.py /home/ubuntu/rl_learner/
COPY Learner.py /home/ubuntu/rl_learner/
COPY Learner_v1.py /home/ubuntu/rl_learner/
COPY params.pkl /home/ubuntu/rl_learner/
COPY RLbrain_v1.py /home/ubuntu/rl_learner/
COPY socketserver.py /home/ubuntu/rl_learner/
COPY test.py /home/ubuntu/rl_learner/

# Only works in Kubernetes
CMD ["bash", "-c", "python3 Learner_v1.py --host $RL_LEARNER_SERVICE_HOST --port $RL_LEARNER_SERVICE_PORT"]

EXPOSE 30007