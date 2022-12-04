import os
from functools import partial
from os.path import join
from pathlib import Path
from typing import Optional, Union, List, Sequence

import numpy as np
import xarray as xr
from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.torch_datasets import get_tensor_dataset_from_numpy
from src.utilities.utils import get_logger

log = get_logger(__name__)


def drop_lat_lon_info(ds, time_slice=None):
    """ Drop any geographical metadata for lat/lon so that xarrays are
     concatenated along example/grid-box instead of lat/lon dim. """
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    return ds.assign_coords(lat=np.arange(ds.sizes['lat']), lon=np.arange(ds.sizes['lon']))


class OISSTv2DataModule(BaseDataModule):
    def __init__(self,
                 data_dir: str,
                 boxes: Union[List, str] = 'all',
                 predict_boxes: Union[List, str] = 'all',
                 predict_slice: Optional[slice] = slice('2020-12-01', '2020-12-31'),
                 box_size: int = 60,
                 horizon: int = 1,
                 pixelwise_normalization: bool = True,
                 **kwargs):
        if 'oisst' not in data_dir:
            if os.path.isdir(join(data_dir, 'oisstv2-daily')):
                data_dir = join(data_dir, 'oisstv2-daily')
            elif os.path.isdir(join(data_dir, 'oisst-daily')):
                data_dir = join(data_dir, 'oisst-daily')

        suffix = 'pixelwise' if pixelwise_normalization else 'box_mean'
        data_dir = join(data_dir, f'subregion-{box_size}x{box_size}boxes-{suffix}_stats')
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        # Set the temporal slices for the train, val, and test sets
        self.train_slice = slice(None, '2018-12-31')
        self.val_slice = slice('2019-01-01', '2019-12-31')
        self.test_slice = slice('2020-01-01', '2021-12-31')
        self.stage_to_slice = {'fit': slice(self.train_slice.start, self.val_slice.stop),
                               'validate': self.val_slice,
                               'test': self.test_slice, 'predict': predict_slice,
                               None: None}
        log.info(f"Using OISSTv2 data directory: {self.hparams.data_dir}")

    def _check_args(self):
        boxes = self.hparams.boxes
        assert self.hparams.horizon > 0, f"horizon must be > 0, but is {self.hparams.horizon}"
        assert self.hparams.box_size > 0, f"box_size must be > 0, but is {self.hparams.box_size}"
        assert isinstance(boxes, Sequence) or boxes in [
            'all'], f"boxes must be a list or 'all', but is {self.hparams.boxes}"

    def get_glob_pattern(self, boxes: Union[List, str] = 'all'):
        ddir = Path(self.hparams.data_dir)
        if isinstance(boxes, Sequence) and boxes != 'all':
            return [ddir / f'sst.day.mean.box{b}.nc' for b in boxes]
        elif boxes == 'all':
            return str(ddir / 'sst.day.mean.box*.nc')  # os.listdir(self.hparams.data_dir)
        else:
            raise ValueError(f'Unknown value for boxes: {boxes}')

    def update_predict_data(self,
                            boxes: Union[List, str] = 'all',
                            predict_slice: Optional[slice] = slice('2020-12-01', '2020-12-31')):
        self.hparams.predict_boxes = boxes
        self.stage_to_slice['predict'] = predict_slice

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        # Read all boxes into a single xarray dataset (slice out all innecessary time steps)
        preprocess = partial(drop_lat_lon_info, time_slice=self.stage_to_slice[stage])
        if stage == 'predict':
            glob_pattern = self.get_glob_pattern(self.hparams.predict_boxes)
        else:
            glob_pattern = self.get_glob_pattern(self.hparams.boxes)
        import dask
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            try:
                ds = xr.open_mfdataset(
                    glob_pattern,
                    combine='nested', concat_dim='grid_box', preprocess=preprocess
                ).sst
            except OSError as e:
                log.error(f"Could not open OISSTv2 data files from {glob_pattern}."
                          f" Check that the data directory is correct: {self.hparams.data_dir}")
                raise e
            # Set the correct tensor datasets for the train, val, and test sets
            ds_train = ds.sel(time=self.train_slice) if stage in ['fit', None] else None
            ds_val = ds.sel(time=self.val_slice) if stage in ['fit', 'validate', None] else None
            ds_test = ds.sel(time=self.test_slice) if stage in ['test', None] else None
            ds_predict = ds.sel(time=self.stage_to_slice['predict']) if stage == 'predict' else None

            ds_splits = {'train': ds_train, 'val': ds_val, 'test': ds_test, 'predict': ds_predict}
            for split, split_ds in ds_splits.items():
                if split_ds is None:
                    continue
                # Split ds into input and target (which is horizon time steps ahead of input, X)
                X = split_ds.isel(time=slice(None, -self.hparams.horizon))
                Y = split_ds.isel(time=slice(self.hparams.horizon, None))
                # X and Y are 4D tensors with dimensions (grid-box, time, lat, lon)

                # Merge the time and grid_box dimensions into a single example dimension, and reshape
                X = X.stack(example=('time', 'grid_box')).transpose('example', 'lat', 'lon').values
                Y = Y.stack(example=('time', 'grid_box')).transpose('example', 'lat', 'lon').values
                # X and Y are now 3D tensors with dimensions (example, lat, lon)

                # Add dummy channel dimension to first axis (1 channel, since we have only one variable, SST)
                X, Y = np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1)
                # X and Y are now 4D tensors with dimensions (example, channel, lat, lon), where channel=1

                # Create the pytorch tensor dataset
                tensor_ds = get_tensor_dataset_from_numpy(X, Y, dataset_id=split)

                # Save the tensor dataset to self._data_{split}
                setattr(self, f'_data_{split}', tensor_ds)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)
