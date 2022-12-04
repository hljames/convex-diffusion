# Environment Setup
This project is developed in Python 3.9.
You can choose to use a conda/virtual environment, 
or use the provided docker image (available in [Dockerhub](https://hub.docker.com/repository/docker/salv4/climateml).

## Conda environment
There are multiple options for which environment to use depending on your use case.
In the following commands, you can use a different environment name,
by changing string that comes after the ``-n`` flag in the ``conda env create ...`` lines

### Training (and evaluating)
#### On GPU
    conda env create -f env_train_gpu.yaml -n MY_ENV_NAME
    conda activate MY_ENV_NAME  # activate the environment called <MY_ENV_NAME>

#### On CPU
    conda env create -f env_train_cpu.yaml -n MY_ENV_NAME
    conda activate MY_ENV_NAME  


### Only evaluate and do inference 
The following environment file is for inference only, i.e. if you want to
evaluate (or predict with) a model that has already been trained.

    conda env create -f env_evaluation.yaml -n climateml-eval
    conda activate climateml-eval  

## Note for jupyter notebooks: 

You need to choose the above environment (e.g. ``climateml-eval``) as kernel of the jupyter notebook.
If the environment doesn't show up in the list of possible kernels, please do
    
    python -m ipykernel install --user --name climateml-eval   # change aibedo with whatever environment name you use 

Then, please refresh the notebook page.


## Docker container
See the [build-push-docker-image](../scripts/build-push-docker-image.sh) 
script for how to build and push the docker image to {Dockerhub, Nautilus registry}.
