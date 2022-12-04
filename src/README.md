# Using the code-base
**IMPORTANT NOTE:** 
All commands in this README assume that you are in the [root of the repository](../) (and need to be run from there)!

## Train a model
From the [repository root](../run.py), please run the following in the command line:    

    python run.py trainer.gpus=0 model=unet_convnext logger=none callbacks=default

This will train a UNet ConvNext on the CPU using some default callbacks and hyperparameters, but no logging.

## Running experiments
It is recommended to define all the parameters of your experiment 
in a [YAML](https://yaml.org/) file inside the [configs/experiment](configs/experiment) folder.

For example, the config file [oisst_full_unet_convnext](configs/experiment/oisst_full_unet_convnext.yaml) defines
the experiment to train a UNet ConvNext on the full OISSTv2 dataset with some particular (hyper-)parameters.
You can then easily run such an experiment with the following command:

    python run.py experiment=oisst_full_unet_convnext    # replace 'oisst_unet with the name of your config file

## Resume training from a wandb run
If you want to resume training from a previous run, you can use the following command:

    python run.py logger.wandb.id=<run_id>  

where `<run_id>` is the wandb ID of the run you want to resume training.  
You can add any extra arguments, e.g. ``datamodule.num_workers=8``, to change the values from the previous run.
Note that if you want to run for more epoch, you need to add ``trainer.max_epochs=<new_max_epochs>``.

## Important Training Arguments and Hyperparameters
- To run on CPU use ``trainer.gpus=0``, to use a single GPU use ``trainer.gpus=1``, etc.
- To override the data directory you can override the flag ``datamodule.data_dir=<data-dir>``, or see the [data README](../data/README.md) for more options.
- A random seed for reproducibility can be set with ``seed=<seed>`` (by default it is ``11``).

### Directories for logging and checkpoints
By default,
- the checkpoints (i.e. model weights) are saved in ``results/checkpoints/``,
- any logs are saved in ``results/logs/``.

To change the name of ``results/`` in both subdirs above, you may simply use the flag ``work_dir=YOUR-OUT-DIR``.
To only change the name of the checkpoints directory, you may use the flag ``ckpt_dir=YOUR-CHECKPOINTS-DIR``.
To only change the name of the logs directory, you may use the flag ``log_dir=YOUR-LOGS-DIR``.

### Debugging
To run the code in debug mode, use the flag ``mode=debug``. 
To debug the OISSTv2 data/models, use ``mode=debug_oisst``,
and have the subregion boxes 0, 1, 133 downloaded (since the debug run only uses these boxes for fast data loading).

### Data parameters and structure
#### General data-specific parameters
Important data-specific parameters can be all found in the 
[configs/datamodule/base_data_config](configs/datamodule/_base_data_config.yaml) file. 
In particular:
- ``datamodule.data_dir``: the directory where the data must be stored (see the [data README](../data/README.md) for more details).
- ``datamodule.batch_size``: the batch size to use for training.
- ``datamodule.num_workers``: the number of workers to use for loading the data.

You can override any of these parameters by adding ``datamodule.<parameter>=<value>`` to the command line.

### ML model parameters and architecture

#### Define the architecture
To train a pre-defined model do ``model=<model_name>``, e.g. ``model=cnn``, ``model=mlp``, etc.,
    where [configs/model/](configs/model)<model_name>.yaml must be the configuration file for the respective model.

You can also override any model hyperparameter by adding ``model.<hyperparameter>=<value>`` to the command line.
E.g.:
- to change the number of layers and dimensions in an MLP you would use 
``model=mlp 'model.hidden_dims=[128, 128, 128]'`` (note that the parentheses are needed when the value is a list).
- to change the MLP ratio in the AFNO model you would use ``model=transformer 'model.mlp_ratio=0.5'``.

#### General model-specific parameters
Important model-specific parameters can be all found in the 
[configs/model/_base_model_config](configs/model/_base_model_config.yaml) file. 
In particular:
- ``model.scheduler``: the scheduler to use for the learning rate. Default: Exponential decay with gamma=0.98
- ``model.monitor``: the logged metric to track for early-stopping, model-checkpointing and LR-scheduling. Default: ``val/mse``.

### Hyperparameter optimization
Hyperparameter optimization is supported via the Optuna Sweeper.
Please read the instructions for setting it up and running experiments with Optuna in the
[Optuna configs README](configs/optuna/README.md).


### Wandb support
<details>
  <summary><b> Requirements & Logging in </b></summary>
The following requires you to have a wandb (team) account, and you need to login with ``wandb login`` before you can use it.
You can also simply export the environment variable ``WANDB_API_KEY`` with your wandb API key, 
and the [run.py](../run.py) script will automatically login for you.

</details>

- To log metrics to [wandb](https://wandb.ai/site) use ``logger=wandb``.
- To use some nice wandb specific callbacks in addition, use ``callbacks=wandb`` (e.g. save the best trained model to the wandb cloud).

## Tips

<details>
    <summary><b> hydra.errors.InstantiationException </b></summary>

The ``hydra.errors.InstantiationException`` itself is not very informative, 
so you need to look at the preceding exception(s) (i.e. scroll up) to see what went wrong.
</details>

<details>
    <summary><b> Overriding nested Hydra config groups </b></summary>

Nested config groups need to be overridden with a slash - not with a dot, since it would be interpreted as a string otherwise.
For example, if you want to change the filter in the AFNO transformer:
``python run.py model=afno model/mixer=self_attention``
And if you want to change the optimizer, you should run:
``python run.py  model=graphnet  optimizer@model.optimizer=SGD``
</details>

<details>
  <summary><b> Local configurations </b></summary>

You can easily use a local config file (that,e.g., overrides data dirs, working dir etc.), by putting such a yaml config 
in the [configs/local/](configs/local) subdirectory (Hydra searches for & uses by default the file configs/local/default.yaml, if it exists)
</details>

<details>
    <summary><b> Wandb </b></summary>

If you use Wandb, make sure to select the "Group first prefix" option in the panel/workspace settings of the web app inside the project (in the top right corner).
This will make it easier to browse through the logged metrics.
</details>




