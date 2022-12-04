# Nautilus cluster
## Pre-requisites

First of all, you need to have Nautilus access 
[as described in the wiki](https://github.com/Rose-STL-Lab/Lab-Wiki/wiki/Servers#access-ucsd-nautilus-clusters-preferred).

### GitHub Integration
To run the most up-to-date Github code directly on Nautilus please
[create a Github access token](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line)
and then save it (and your github username) to kubectl as follows


    kubectl create secret generic my-github-secret --from-literal=user=<USERNAME> --from-literal=token=<ACCESS-TOKEN>

**Important:** ``my-github-secret`` is the name of the secret, and can be changed to anything you like. 
The same name must be used in the ``.yaml`` file (in ``initContainers/env/valueFrom/secretKeyRef/ref``).
## Running the code
To edit the commands that are run on the cluster, edit the [run.yaml](nautilus/run.yaml) file (the lines corresponding to ``containers/args``).
When you are ready to run the code and kubectl config file, you can run it on Nautilus as follows

    kubectl create -f nautilus/run.yaml

By default, the above config file simply calls [run.py](run.py).

## Interactive sessions
To run an interactive session on Nautilus, you can use the [interactive.yaml](nautilus/interactive.yaml) file, i.e. run
    
        kubectl create -f interactive-pod.yaml

Then, you can connect to the created pod via

        kubectl exec -it interactive-pod -- bash


## Weights and Biases (wandb) on Nautilus
To log with wandb, this worked for me (and each of the steps were important to avoid errors):
- Export your wandb API key to your environment: ``export WANDB_API_KEY=<YOUR_API_KEY>``
(or run your script with ``WANDB_API_KEY=<YOUR_API_KEY> python run.py``)
- Set ``export WANDB_CONFIG_DIR=<SOME_WRITABLE_DIR>`` (e.g. the working directory that you are using)


## JupyterHub Integration
On the Jupyerhub Nautilus West, you can access the OISSTv2 data in the ``/cephfs`` directory.

Run the following command to create a JupyterHub pod on Nautilus

    kubectl create -f nautilus/jupyterhub-pod.yaml

The starting command ``jupyter lab`` in [jupyterhub-pod.yaml](nautilus/jupyterhub-pod.yaml)
initiates a jupyter server at port 8888 inside the container. 
The trick here is to map the port 8888 from container to your desktop using 

    kubectl port-forward jupyterhub 8888:8888 --address "0.0.0.0"


Notes:
- The address 0.0.0.0 is necessary for the server to accept your IP’s 
(different from container) connection ... otherwise you can only access it from inside
- Make sure you do not have a program/process that already occupies port 8888 
(if you are using unix-like system, simply use ``lsof -i :8888`` to check for such a program). 
If so, modify the destination port to another port (like 8888:8889).
- Then you should be able to access the jupyterhub in your browser with address ``localhost:8888``. 
- The token defaults to ``627a7b3b`` in the yaml file, you may modify it.
- With the yaml file above, you are able to mount any persistent storage (pvc) to it.
- An alternative way is to still use ``sleep infinity`` to start the node, 
enter the terminal, and manually run ``jupyter lab``. 
If you are using VSCode, it may automatically detect you start a new program and run ``port-forward`` for you.


## Useful commands

    kubectl get pods   # shows you the runs with their hashes 
    kubectl logs <RUN-HASH>

To delete multiple pods at once do

    kubectl get pods -o name --no-headers=true | grep  ... | xargs -n 1 kubectl delete pod

Note that the pods from jobs are automagically removed in 2 days, 
and you can adjust the ``ttlSecondsAfterFinished`` field to remove them faster.

