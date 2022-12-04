FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.6.1
LABEL maintainer="Salva Ruhling Cachay <sruhlingcachay@ucsd.edu>"

# ----------------------------------------------------------------
# This image is available at:
# https://hub.docker.com/repository/docker/salv4/climate-ml
# ----------------------------------------------------------------

USER $NB_USER
ENV USER=${NB_USER}

RUN apt-get update \
    && apt-get install -yq --no-install-recommends vim emacs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install pytorch-lightning==1.7.7 torchmetrics xarray netCDF4 dask einops hydra-core wandb plotly optuna
COPY hydra_optuna_sweeper/ /opt/app
WORKDIR /opt/app
# Install hydra_optuna_sweeper
RUN pip install .