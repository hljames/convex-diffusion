from omegaconf import DictConfig


def get_dims_of_dataset(datamodule_config: DictConfig):
    """ Returns the number of features for the given dataset. """
    target = datamodule_config.get('_target_', datamodule_config.get('name'))
    if 'oisstv2' in target:
        box_size = datamodule_config.box_size
        input_dim, output_dim, spatial_dims = 1, 1, (box_size, box_size)
    else:
        raise ValueError(f"Unknown dataset: {target}")
    return {'input': input_dim, 'output': output_dim, 'spatial': spatial_dims}