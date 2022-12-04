from typing import Any, Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.models._base_model import BaseModel


class BaseDiffusion(BaseModel):
    def __init__(
            self,
            model=None,
            **kwargs
    ):
        if model is None:
            raise ValueError(f'Arg ``model`` is missing...'
                             f' Please provide a backbone model for the diffusion model (e.g. a Unet)')
        base_kwargs = {k: model.hparams.get(k)
                       for k in ['monitor', 'mode', 'loss_function', 'optimizer', 'scheduler', 'verbose']}
        base_kwargs['datamodule_config'] = model.datamodule_config if hasattr(model, 'datamodule_config') else None
        # override base_kwargs with kwargs
        base_kwargs.update(kwargs)
        super().__init__(**base_kwargs)
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        self.image_size = self.spatial_dims[0]
        # self.num_timesteps = int(timesteps)

        self.example_input_array = model.example_input_array if hasattr(model, 'example_input_array') else None

    def sample(self, condition=None, num_samples=1, **kwargs):
        # sample from the model
        raise NotImplementedError()

    def predict(self, X, **kwargs):
        return self.sample(condition=X, **kwargs)

    def forward(self, batch, *args, **kwargs):
        inputs, targets = batch
        b, c, h, w = targets.shape
        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

        # img = normalize_to_neg_one_to_one(img)
        return self.p_losses(targets, condition=inputs, t=t, *args, **kwargs)

    def get_loss(self, batch, *args, **kwargs) -> Tensor:
        return self(batch, *args, **kwargs)

    def _evaluation_step(self, *args, **kwargs):
        if 'batch_idx' in kwargs.keys() and kwargs['batch_idx'] not in self.val_batch_idxs:
            return None
        return super()._evaluation_step(*args, **kwargs, on_step=True, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # Only go over random subset of validation set
        n_batches = len(self.trainer.datamodule.val_dataloader())
        if n_batches < 10000:
            use_batches = n_batches // 3
        else:
            use_batches = 5000
        self.log_text.info(f'Using {use_batches} batches for validation')
        # print(f"n_batches in validation: {n_batches}")
        self.val_batch_idxs = np.random.choice(n_batches, size=use_batches, replace=False)
