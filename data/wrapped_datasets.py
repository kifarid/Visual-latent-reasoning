from .dataset_cache_wrapper_h5 import cacheable_dataset
from .custom_multiframe import MultiHDF5DatasetMultiFrame


# wrapped_datasets.py


# Wrap datasets using WrapperH5
CachedMultiHDF5DatasetMultiFrame = cacheable_dataset(MultiHDF5DatasetMultiFrame)

