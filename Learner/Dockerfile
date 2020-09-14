FROM ubuntu:18.04

RUN apt update && apt install -y python3-pip
COPY requirements.txt /home/ubuntu/Learner/
WORKDIR /home/ubuntu/Learner/
RUN pip3 install -r requirements.txt

COPY __init__.py /home/ubuntu/Learner/
COPY learner.py /home/ubuntu/Learner/
COPY params.pkl /home/ubuntu/Learner/
COPY vtrace_tf.py /home/ubuntu/Learner/
COPY vtrace.py /home/ubuntu/Learner/
COPY socketserver.py /home/ubuntu/Learner/
COPY agent.py /home/ubuntu/Learner/

# Only works in Kubernetes
CMD ["bash", "-c", "python3 Learner_v1.py --host $LEARNER_SERVICE_HOST --port $LEARNER_SERVICE_PORT"]

EXPOSE 30007