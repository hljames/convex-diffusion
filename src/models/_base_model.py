import logging
import time
from typing import Optional, List, Any, Dict, Sequence, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor, nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torchmetrics
import wandb

from src.datamodules.dataset_dimensions import get_dims_of_dataset
from src.utilities.utils import get_logger, to_DictConfig, get_loss, raise_error_if_invalid_value


class BaseModel(LightningModule):
    r""" This is a template base class, that should be inherited by any stand-alone ML model.
    Methods that need to be implemented by your concrete ML model (just as if you would define a :class:`torch.nn.Module`):
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        >>> self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7


    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model())!

    Args:
        datamodule_config: DictConfig with the configuration of the datamodule
        optimizer: DictConfig with the optimizer configuration (e.g. for AdamW)
        scheduler: DictConfig with the scheduler configuration (e.g. for CosineAnnealingLR)
        monitor (str): The name of the metric to monitor, e.g. 'val/mse'
        mode (str): The mode of the monitor. Default: 'min' (lower is better)
        loss_function (str): The name of the loss function. Default: 'mean_squared_error'
        name (str): optional string with a name for the model
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 datamodule_config: DictConfig = None,
                 optimizer: Optional[DictConfig] = None,
                 scheduler: Optional[DictConfig] = None,
                 monitor: Optional[str] = None,
                 mode: str = "min",
                 loss_function: str = "mean_squared_error",
                 name: str = "",
                 verbose: bool = True,
                 ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.monitor
        self.save_hyperparameters(ignore=['datamodule_config', 'verbose', 'model'])
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__ if name == '' else name)
        self.name = name
        self.verbose = verbose
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        # Infer the data dimensions
        if datamodule_config is None:
            raise ValueError(f'You need to pass a datamodule_config to the constructor of {self.__class__.__name__},'
                             f' which includes attributes such as ``data_dir``, ``_target_``, etc.')
        self.datamodule_config = datamodule_config
        self._data_dir = datamodule_config.data_dir
        dims = get_dims_of_dataset(datamodule_config)
        self.num_input_channels = dims['input']
        self.num_output_channels = dims['output']
        self.spatial_dims = dims['spatial']

        # Get the loss function
        self.criterion = get_loss(loss_function)

        # Timing variables to track the training/epoch/validation time
        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None

        # Metrics
        val_metrics = {'val/mse': torchmetrics.MeanSquaredError(squared=True)}
        self.val_metrics = nn.ModuleDict(val_metrics)
        self._test_metrics = self._predict_metrics = None
        # Check that the args/hparams are valid
        self._check_args()

    @property
    def test_set_name(self) -> str:
        return self.trainer.datamodule.test_set_name if hasattr(self.trainer.datamodule, 'test_set_name') else 'test'

    @property
    def prediction_set_name(self) -> str:
        return self.trainer.datamodule.prediction_set_name if hasattr(self.trainer.datamodule,
                                                                      'prediction_set_name') else 'predict'

    @property
    def test_metrics(self):
        if self._test_metrics is None:
            metrics = {f'{self.test_set_name}/mse': torchmetrics.MeanSquaredError(squared=True)}
            self._test_metrics = nn.ModuleDict(metrics).to(self.device)
        return self._test_metrics

    @property
    def predict_metrics(self):
        if self._predict_metrics is None:
            metrics = {f'{self.prediction_set_name}/rmse': torchmetrics.MeanSquaredError(squared=False),
                       f'{self.prediction_set_name}/mae': torchmetrics.MeanAbsoluteError()}
            self._predict_metrics = nn.ModuleDict(metrics).to(self.device)
        return self._predict_metrics

    @property
    def n_params(self):
        """ Returns the number of parameters in the model """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def data_dir(self) -> str:
        if self._data_dir is None:
            self._data_dir = self.trainer.datamodule.hparams.data_dir
        return self._data_dir

    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    def forward(self, X: Tensor):
        r""" Standard ML model forward pass (to be implemented by the specific ML model).

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
        Shapes:
            - Input: :math:`(B, *, C_{in})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` is the number of input features/channels.
        """
        raise NotImplementedError('Base model is an abstract class!')

    def predict(self, X: Tensor, **kwargs) -> Any:
        """
        This should be the main method to use for making predictions/doing inference.

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`.
                This is the same tensor one would use in :func:`forward`.
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Tensor]: The model predictions (in a post-processed format), i.e. a dictionary output_var -> output_var_prediction,
                where each output_var_prediction is a Tensor of shape :math:`(B, *)` in original-scale (e.g.
                in Kelvin for temperature), and non-negativity has been enforced for variables such as precipitation.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: Dict :math:`k_i` -> :math:`v_i`, and each :math:`v_i` has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features.
        """
        return self(X)  # by default, just call the forward method

    # --------------------- training with PyTorch Lightning
    def on_train_start(self) -> None:
        """ Log some info about the model/data at the start of training """
        self.log('Parameter count', float(self.n_params))
        self.log('Training set size', float(len(self.trainer.datamodule._data_train)))
        n_val_samples = len(self.trainer.datamodule._data_val)
        n_val_batches = n_val_samples // self.trainer.datamodule.hparams.eval_batch_size
        # print(f'Number of validation batches: {self.n_val_batches}')
        self.log('Validation set size', float(n_val_samples))

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()
        torch.cuda.reset_peak_memory_stats(self.trainer.root_gpu)
        torch.cuda.synchronize(self.trainer.root_gpu)

    def train_step_initial_log_dict(self) -> dict:
        return dict()

    def get_loss(self, batch: Any, *args, **kwargs) -> Tensor:
        r""" Compute the loss for the given batch
        Concrete implementations may override this method for special forward passes (e.g. with multiple outputs).
        """
        X, Y = batch
        # Predict
        preds = self(X)
        # Compute loss
        loss = self.criterion(preds, Y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        r""" One step of training (backpropagation is done on the loss returned at the end of this function) """
        train_log = self.train_step_initial_log_dict()
        # Compute main loss
        loss = self.get_loss(batch)

        # Logging of train loss and other diagnostics
        train_log["train/loss"] = loss.item()

        # Count number of zero gradients as diagnostic tool
        train_log['n_zero_gradients'] = sum(
            [int(torch.count_nonzero(p.grad == 0))
             for p in self.parameters() if p.grad is not None
             ]) / self.n_params

        self.log_dict(train_log, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        train_time = time.time() - self._start_epoch_time
        torch.cuda.synchronize(self.trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(self.trainer.root_gpu) / 2 ** 20
        self.log_dict({'epoch': float(self.current_epoch), "time/train": train_time})

        try:
            # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
            max_memory = self.trainer.training_type_plugin.reduce(max_memory)
            epoch_time = self.trainer.training_type_plugin.reduce(train_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

    # --------------------- evaluation with PyTorch Lightning
    def _evaluation_step(self,
                         batch: Any,
                         batch_idx: int,
                         torch_metrics: Optional[nn.ModuleDict] = None,
                         log_kwargs: dict = None,
                         on_step: bool = False,
                         on_epoch: bool = True,
                         **kwargs
                         ):
        X, Y = batch
        preds = self.predict(X, **kwargs)
        log_dict1 = dict()

        log_kwargs = log_kwargs or dict()
        log_kwargs['sync_dist'] = True     # for DDP training
        # Compute metrics
        for metric_name, metric in torch_metrics.items():
            metric(preds, Y)  # compute metrics (need to be in separate line to the following line!)
            log_dict1[metric_name] = metric
        self.log_dict(log_dict1, on_step=on_step, on_epoch=on_epoch, **log_kwargs)  # log metric objects

        return {'targets': Y, 'preds': preds}

    def _evaluation_get_preds(self, outputs: List[Any]) -> Dict[str, np.ndarray]:
        targets = torch.cat([batch['targets'] for batch in outputs], dim=0).cpu().numpy()
        preds = torch.cat([batch['preds'] for batch in outputs], dim=0).detach().cpu().numpy()
        return {'targets': targets, 'preds': preds}

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        log_kwargs = dict(prog_bar=True)
        results = self._evaluation_step(batch, batch_idx, torch_metrics=self.val_metrics, log_kwargs=log_kwargs, verbose=False)
        return results

    def validation_epoch_end(self, outputs: List[Any]) -> dict:
        val_time = time.time() - self._start_validation_epoch_time
        val_stats = {"time/validation": val_time}
        # validation_outputs = self._evaluation_get_preds(outputs)
        # Y_val, validation_preds = validation_outputs['targets'], validation_outputs['preds']

        # target_val_metric = val_stats.pop(self.hparams.monitor, None)
        self.log_dict({**val_stats, 'epoch': float(self.current_epoch)}, prog_bar=False)
        # Show the main validation metric on the progress bar:
        # self.log(self.hparams.monitor, target_val_metric, prog_bar=True)
        return val_stats

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = time.time()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        results = self._evaluation_step(batch, batch_idx, torch_metrics=self.test_metrics)
        return results

    def test_epoch_end(self, outputs: List[Any]):
        test_time = time.time() - self._start_test_epoch_time
        self.log("time/test", test_time)

    def on_predict_start(self) -> None:
        assert self.trainer.datamodule._data_predict is not None, "_data_predict is None"
        assert self.trainer.datamodule._data_predict.dataset_id == 'predict', "dataset_id is not 'predict'"
        for pdl in self.trainer.predict_dataloaders:
            assert pdl.dataset.dataset_id == 'predict', f"dataset_id is not 'predict', but {pdl.dataset.dataset_id}"

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs
                     ) -> Dict[str, Dict[str, Tensor]]:
        X, Y = batch
        raise NotImplementedError("Not yet implemented!")

    def on_predict_end(self, results: List[Any] = None) -> None:
        if wandb.run is not None:
            log_dict = {'epoch': float(self.current_epoch)}
            for k, v in self.predict_metrics.items():
                log_dict[k] = float(v.compute().item())
                v.reset()
            self.log_text.info(log_dict)
            wandb.log(log_dict)
        else:
            self.log_text.warning("Wandb is not initialized, so no predictions are logged")

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    def _get_optim(self, optim_name: str, **kwargs):
        """
        Method that returns the torch.optim optimizer object.
        May be overridden in subclasses to provide custom optimizers.
        """
        return torch.optim.AdamW(self.parameters(), **kwargs)
        # from timm.optim import create_optimizer_v2
        # return create_optimizer_v2(model_or_params=self, opt=optim_name, **kwargs)

    def configure_optimizers(self):
        """ Configure optimizers and schedulers """
        if 'name' not in to_DictConfig(self.hparams.optimizer).keys():
            self.log_text.info(" No optimizer was specified, defaulting to AdamW with 1e-4 lr.")
            self.hparams.optimizer.name = 'adamw'

        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k not in ['name', '_target_']}
        optimizer = self._get_optim(self.hparams.optimizer.name, **optim_kwargs)
        # Build the scheduler
        if self.hparams.scheduler is None:
            return optimizer  # no scheduler
        else:
            if '_target_' not in to_DictConfig(self.hparams.scheduler).keys():
                raise ValueError("Please provide a _target_ class for model.scheduler arg!")
            scheduler_params = to_DictConfig(self.hparams.scheduler)
            scheduler = hydra.utils.instantiate(scheduler_params, optimizer=optimizer)

        if not hasattr(self.hparams, 'monitor') or self.hparams.monitor is None:
            self.hparams.monitor = f'val/mse'
        if not hasattr(self.hparams, 'mode') or self.hparams.mode is None:
            self.hparams.mode = 'min'

        lr_dict = {'scheduler': scheduler, 'monitor': self.hparams.monitor}  # , 'mode': self.hparams.mode}
        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}

    # Unimportant methods
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def register_buffer_dummy(self, name, tensor, **kwargs):
        try:
            self.register_buffer(name, tensor, **kwargs)
        except TypeError:  # old pytorch versions do not have the arg 'persistent'
            self.register_buffer(name, tensor)
