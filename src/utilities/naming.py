from typing import Union, List, Dict, Optional
from omegaconf import DictConfig


def _shared_prefix(config: DictConfig, init_prefix: str = "") -> str:
    """ This is a prefix for naming the runs for a more agreeable logging."""
    s = init_prefix if isinstance(init_prefix, str) else ""
    # Find mixer type if it is a transformer model (e.g. self-attention or FNO mixing)
    kwargs = dict(mixer=config.model.mixer._target_) if config.model.get('mixer') else dict()
    s += clean_name(config.model._target_, **kwargs)
    return s.lstrip('_')


def get_name_for_hydra_config_class(config: DictConfig) -> Optional[str]:
    """ Will return a string that can describe the class of the (sub-)config."""
    if 'name' in config and config.get('name') is not None:
        return config.get('name')
    elif '_target_' in config:
        return config._target_.split('.')[-1]
    return None


def get_detailed_name(config, add_prefix: bool = True) -> str:
    """ This is a detailed name for naming the runs for logging."""
    s = config.get("name") + '_' if config.get("name") is not None else ""
    s += config.get('name_suffix') + '_' if config.get('name_suffix') is not None else ""
    s += _shared_prefix(config) + '_'

    hdims = config.model.get('hidden_dims')
    if hdims is None:
        num_L = config.model.get('num_layers') or config.model.get('depth')
        if num_L is None:
            num_L = len(config.model.get('dim_mults')) if config.model.get('dim_mults') else '?'
        hdim = config.model.get('hidden_dim') or config.model.get('dim')
        if hdim is not None:
            hdims = f"{hdim}x{num_L}"
    elif all([h == hdims[0] for h in hdims]):
        hdims = f"{hdims[0]}x{len(hdims)}"
    else:
        hdims = str(hdims)
    s += f"{hdims}h_"
    s += f"{config.model.optimizer.get('lr')}lr_"
    if config.model.optimizer.get('weight_decay') and config.model.optimizer.get('weight_decay') > 0:
        s += f"{config.model.optimizer.get('weight_decay')}wd_"

    s += f"{config.get('seed')}seed"
    return s.replace('None', '')


def clean_name(class_name, mixer=None, dm_type=None) -> str:
    """ This names the model class paths with a more concise name."""
    if "AFNONet" in class_name or 'Transformer' in class_name:
        if mixer is None or "AFNO" in mixer:
            s = 'FNO'
        elif "SelfAttention" in mixer:
            s = 'self-attention'
        else:
            raise ValueError(class_name)
    elif 'UnetConvNext' in class_name:
        s = 'UnetConvNext'
    elif 'SimpleChannelOnlyMLP' in class_name:
        s = 'SiMLP'
    elif "MLP" in class_name:
        s = 'MLP'
    elif "Unet" in class_name:
        s = 'UNetResNet'
    elif 'SimpleConvNet' in class_name:
        s = 'SimpleCNN'
    elif "graph_network" in class_name:
        s = 'GraphNet'
    elif "CNN_Net" in class_name:
        s = 'CNN'
    else:
        raise ValueError(f'Unknown class name: {class_name}, did you forget to add it to the clean_name function?')

    return s


def get_group_name(config) -> str:
    """
    This is a group name for wandb logging.
    On Wandb, the runs of the same group are averaged out when selecting grouping by `group`
    """
    s = get_name_for_hydra_config_class(config.model)
    s = s or _shared_prefix(config, init_prefix=s)
    return s


def var_names_to_clean_name() -> Dict[str, str]:
    """ This is a clean name for the variables (e.g. for plotting)"""
    var_dict = {
        'tas': 'Air Temperature',
        'psl': "Sea-level Pressure",
        'ps': "Surface Pressure",
        'pr': "Precipitation",
        'sst': "Sea Surface Temperature",
    }
    return var_dict
