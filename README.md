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