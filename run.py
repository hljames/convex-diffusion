import os
import hydra
import torch
from omegaconf import DictConfig
from src.train import run_model


@hydra.main(config_path="src/configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """ Run/train model based on the config file configs/main_config.yaml (and any command-line overrides). """
    return run_model(config)


if __name__ == "__main__":
    if 'WANDB_API_KEY' in os.environ:
        import wandb
        wandb.login(key=os.environ['WANDB_API_KEY'])
    main()
