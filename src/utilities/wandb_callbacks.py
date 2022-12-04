import os
import shutil
import subprocess
from pathlib import Path
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger

from src.utilities.utils import get_logger

log = get_logger(__name__)


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.loggers, list):
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """
    Make wandb watch model at the beginning of the run.
    This will log the gradients of the model (as a histogram for each or some weights updates).
    """

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log_type = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger: WandbLogger = get_wandb_logger(trainer=trainer)
        try:
            logger.watch(model=trainer.model, log=self.log_type, log_freq=self.log_freq, log_graph=True)
        except TypeError as e:
            log.info(
                f" Pytorch-lightning/Wandb version seems to be too old to support 'log_graph' arg in wandb.watch(.)"
                f" Wandb version={wandb.__version__}")
            wandb.watch(models=trainer.model, log=self.log_type, log_freq=self.log_freq)  # , log_graph=True)


class SummarizeBestValMetric(Callback):
    """Make wandb log in run.summary the best achieved monitored val_metric as opposed to the last"""

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger: WandbLogger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        # When using DDP multi-gpu training, one usually needs to get the actual model by .module, and
        # trainer.model.module.module will be the same as pl_module

        model = pl_module # .module if isinstance(trainer.model, DistributedDataParallel) else pl_module
        experiment.define_metric(model.hparams.monitor, summary=model.hparams.mode)


class UploadCheckpointsAsFiles(Callback):
    """Upload best and last checkpoints to wandb as a file."""
    def __init__(self, save_last: bool = True, save_best: bool = True):
        self.save_last = save_last
        self.save_best = save_best

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if not hasattr(trainer, 'checkpoint_callback'):
            log.warning("pl.Trainer has no checkpoint_callback/ModelCheckpoint() callback even though you use"
                        " UploadBestCheckpointAsFile - This callback will be ignored!")

    @rank_zero_only
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        return
        if not hasattr(trainer, 'checkpoint_callback'):
            return
        logger = get_wandb_logger(trainer=trainer)
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            # Copy to local dir and upload
            local_best_ckpt = Path(best_ckpt).name
            shutil.copy(best_ckpt, local_best_ckpt)
            # copy to wandb dir
            shutil.copy(best_ckpt, os.path.join(wandb.run.dir, local_best_ckpt))
            shutil.copy(best_ckpt, os.path.join(wandb.run.dir, 'best.ckpt'))
            logger.experiment.log({'best_model_filepath': local_best_ckpt})
            log.info(f"Best checkpoint path will be saved to wandb from path: {best_ckpt} to {local_best_ckpt}")
            logger.experiment.save(local_best_ckpt)
            os.remove(local_best_ckpt)
        # remove the temporary 'best.ckpt' file that was created by on_epoch_end()
        if os.path.exists('best.ckpt'):
            os.remove('best.ckpt')

    @rank_zero_only
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not hasattr(trainer, 'checkpoint_callback'):
            return
        logger = get_wandb_logger(trainer=trainer)
        last_ckpt = trainer.checkpoint_callback.last_model_path
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if self.save_best and best_ckpt:
            # copy best ckpt to a file called 'best.ckpt' and upload it to wandb
            shutil.copyfile(best_ckpt, 'best.ckpt')
            logger.experiment.save('best.ckpt')
        if self.save_last and last_ckpt:
            logger.experiment.save(last_ckpt)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = True):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        logger.experiment.log_artifact(ckpts)

