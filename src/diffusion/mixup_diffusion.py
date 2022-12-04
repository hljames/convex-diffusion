import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from typing import Any, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange, reduce
from tqdm.auto import tqdm

from src.diffusion._base_diffusion import BaseDiffusion
from src.diffusion.schedules import linear_beta_schedule, cosine_beta_schedule
from src.utilities.utils import default, exists, identity, extract

# small helper modules


# Convex diffusion class
class MixupDiffusion(BaseDiffusion):
    def __init__(
            self,
            timesteps=1000,
            sampling_timesteps=None,
            beta_schedule='cosine',
            interpolation_type: str = 'convex',  # 'convex' or 'ddpm'
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.,
            **kwargs
    ):
        super().__init__(**kwargs)
        # assert not (type(self) == GaussianDiffusion), 'GaussianDiffusion is an abstract class, please use a subclass'
        assert not self.model.hparams.learned_sinusoidal_cond

        self.channels = self.model.channels
        self.conditioned = self.model.hparams.conditioned

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        if self.is_ddim_sampling:
            self.log_text.info(f'using ddim sampling with eta {ddim_sampling_eta}')
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        if interpolation_type == 'convex':
            register_buffer('mixup_multiplier_two', 1 - self.sqrt_alphas_cumprod)
        elif interpolation_type == 'ddpm':
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            register_buffer('mixup_multiplier_two', sqrt_one_minus_alphas_cumprod)
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    @torch.no_grad()
    def sample_loop(self, condition, t=None, x_t=None, **kwargs):
        batch_size = condition.shape[0]
        self.model.eval()
        if t is None:
            t = self.num_timesteps
            x_t = condition

        x_s = x_t  # x_t
        direct_recons = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        while t:
            step_s = torch.full((batch_size,), t - 1, dtype=torch.long).to(device)
            x0_hat = self.model(x_s, time=step_s, condition=condition)  # R(x_s, s) = prediction of target data

            if direct_recons is None:
                direct_recons = x0_hat

            x_s_degraded = x0_hat
            if t != 0:
                # D(x_s, s)
                x_s_degraded = self.q_sample(x_start=x_s_degraded, x_end=condition, t=step_s)

            x_s_sub1_degraded = x0_hat
            if t - 1 != 0:
                # D(x_s, s-1)
                step_s_sub1 = torch.full((batch_size,), t - 2, dtype=torch.long).to(device)
                x_s_sub1_degraded = self.q_sample(x_start=x_s_sub1_degraded, x_end=condition, t=step_s_sub1)

            x = x_s - x_s_degraded + x_s_sub1_degraded
            x_s = x
            t = t - 1

        # self.model.train()
        return x_t, direct_recons, x_s

    def sample(self, condition=None, num_samples=1, **kwargs):
        x_t, direct_recons, x_s = self.sample_loop(condition, **kwargs)
        # x_t is simple the condition (i.e. X_T, for T = # of timesteps)
        # direct_recons is the direct reconstruction of the condition (i.e. R(X_T, T))
        return x_s

    def q_sample(self, x_start, x_end, t, noise=None):
        """ Draw the intermediate degraded data (given the start/target data and the diffused data) """
        # noise = default(noise, lambda: torch.randn_like(x_start))  # create random noise if not provided
        # extract simply returns the alphas for timestep t (in shape of x_start)
        # first term decreases as t increases
        # second term becomes larger as t increases

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.mixup_multiplier_two, t, x_start.shape) * x_end
        )

    def p_losses(self, x_start, condition, t, noise=None):
        """

        Args:
            x_start: the start/target data
            condition: the condition data
            t: the time step of the diffusion process
            noise: the noise to use to sample the diffused data
        """
        b, c, h, w = x_start.shape
        # noise = default(noise, lambda: torch.randn_like(x_start))

        # noised data sample, x_t, where t is the corresponding time step (varies for each batch element)
        x_t = self.q_sample(x_start=x_start, x_end=condition, t=t, noise=noise)  # mixup

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        #x_self_cond = None
        #if self.self_condition and random() < 0.5:
        #    with torch.no_grad():
        #        x_self_cond = self.model_predictions(x, t).pred_x_start
        #        x_self_cond.detach_()

        # predict and take gradient step
        x_start_reconstruction = self.model(x_t, time=t, condition=condition)

        loss = self.criterion(x_start_reconstruction, x_start)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = (loss * extract(self.p2_loss_weight, t, loss.shape)).mean()
        return loss


if __name__ == '__main__':
    # test diffusion model with dummy inputs
    n_channels = 1
    image_size = 32
    from src.models.unet import Unet
    from omegaconf import OmegaConf
    dm_config = {'data_dir': None, 'box_size': image_size, '_target_': 'src.datamodules.oisstv2.OISSTv2DataModule'}
    # to omegaconf
    dm_config = OmegaConf.create(dm_config)
    model = Unet(input_channels=n_channels, dim=64, datamodule_config=dm_config)
    model = MixupDiffusion(timesteps=10,
                           interpolation_type='ddpm',
                           beta_schedule='cosine', loss_function='l2',
                           model=model)
    print(model.sqrt_alphas_cumprod)
    print(model.mixup_multiplier_two)
    x = torch.randn(2, n_channels, image_size, image_size)
    condition = torch.randn(2, n_channels, image_size, image_size)
    batch = (condition, x)
    loss = model(batch)
    print(loss)
