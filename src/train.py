import os.path
import signal
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import src.utilities.config_utils as cfg_utils
from src.interface import get_model_and_data
from src.utilities.utils import get_logger, melk, divein

log = get_logger(__name__)


def run_model(config: DictConfig) -> float:
    r"""
    This function runs/trains/tests the model.

    .. note::
        It is recommended to call this function by running its underlying script, ``aibedo.train.py``,
        as this will enable you to make the best use of the command line integration with Hydra.
        For example, you can easily train an MLP for 10 epochs on the CPU with:

        >>>  python train.py trainer.max_epochs=10 trainer.gpus=0 model=mlp logger=none callbacks=default

    Args:
        config: A DictConfig object generated by hydra containing the model, data, callbacks & trainer configuration.

    Returns:
        float: the best model score reached while training the model.
                E.g. "val/mse", the mean squared error on the validation set.
    """
    # Seed for reproducibility
    seed_everything(config.seed)
    # Reload model checkpoint if needed to resume training
    if config.get('logger') and config.logger.get('wandb') and config.logger.wandb.get('id'):
        from src.utilities.wandb_api import load_hydra_config_from_wandb
        log.info(f"Loading last model weights and config from wandb run {config.logger.wandb.id}")
        run_path = f"{config.logger.wandb.entity}/{config.logger.wandb.project}/{config.logger.wandb.id}"
        # Load correct config from wandb
        extra_args = OmegaConf.from_cli()
        config = load_hydra_config_from_wandb(run_path, override_config=extra_args)  # base_config=config)
        # Load last model checkpoint from wandb
        ckpt_path = wandb.restore('last.ckpt', run_path=run_path, replace=True, root=config.ckpt_dir).name
    else:
        ckpt_path = None

    cfg_utils.extras(config)

    if config.get("print_config"):  # pretty print config yaml -- requires rich package to be installed
        print_fields = ("model", 'diffusion', "datamodule", 'seed', 'work_dir')  # or "all"
        cfg_utils.print_config(config, fields=print_fields)

    # Obtain the instantiated model and data classes from the config
    model, datamodule = get_model_and_data(config)

    # Init Lightning callbacks and loggers (e.g. model checkpointing and Wandb logger)
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, 'callbacks')
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, 'logger')

    # Init Lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, resume_from_checkpoint=ckpt_path  # , deterministic=True
    )

    # Send some parameters from config to be saved by the lightning loggers
    cfg_utils.log_hyperparameters(config=config, model=model, data_module=datamodule, trainer=trainer,
                                  callbacks=callbacks)

    # Save the config to the Wandb cloud (if wandb logging is enabled)
    cfg_utils.save_hydra_config_to_wandb(config)

    if hasattr(signal, 'SIGUSR1'):  # Windows does not support signals
        signal.signal(signal.SIGUSR1, melk(trainer, config.ckpt_dir))
        signal.signal(signal.SIGUSR2, divein(trainer))

    def fit(ckpt_filepath=None):
        N_TRIES = 3
        for trial in range(N_TRIES):
            try:
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_filepath)
                break
            except FileNotFoundError as e:
                if trial == N_TRIES - 1:
                    raise e
                log.warning(f"Error: {e}")
                last_model_ckpt = os.path.join(config.ckpt_dir, 'last.ckpt')
                log.warning(f"Retrying training trial {trial+1}/{N_TRIES}...")
                log.info(f"Loading last model weights from {last_model_ckpt}. Is file? {os.path.isfile(last_model_ckpt)}")
                trainer.fit(model, datamodule=datamodule, ckpt_path=last_model_ckpt)

    try:
        # Train the model
        fit(ckpt_filepath=ckpt_path)
    except Exception as e:
        melk(trainer, config.ckpt_dir)()
        raise e

    # Testing:
    if config.get("test_after_training"):
        trainer.test(datamodule=datamodule, ckpt_path='best')

    if config.get('logger') and config.logger.get("wandb"):
        try:
            wandb.finish()
        except FileNotFoundError as e:
            log.info(f"Wandb finish error:\n{e}")

    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path and False:
        # This is how the best model weights can be reloaded back:
        final_model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            datamodule_config=config.datamodule,
        )

    # return best score (i.e. validation mse). This is useful when using Hydra+Optuna HP tuning.
    return trainer.checkpoint_callback.best_model_score


@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """ Run/train model based on the config file configs/main_config.yaml (and any command-line overrides). """
    return run_model(config)


if __name__ == "__main__":
    main()
