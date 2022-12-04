# Convex Diffusion: Interpolation for Dynamics Forecasting with Diffusion Models

## Environment
Please follow the instructions in the [environment](environment/) folder to set up the correct conda environment.


## Training a model

From the repository root, please run the following in the command line:    

    python run.py trainer.gpus=0 model=unet_convnext logger=none callbacks=default

This will train a UNet ConvNext on the CPU using some default callbacks and hyperparameters, but no logging.
To change the used data directory you can override the flag ``datamodule.data_dir=YOUR-DATA-DIR``.

****For more configuration & training options, please see the [src/README](src/README.md).***

## Compute


- Please follow the instructions in the [nautilus](scripts/nautilus/) folder to set up the repository on the Nautilus cluster.
- Please follow the instructions in the [NERSC](scripts/nersc/) folder to run the repository on the NERSC Perlmutter cluster.


## Data
Please read the [data](data/) folder for instructions on how to download/transfer/set up the data.
