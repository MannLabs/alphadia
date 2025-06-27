#!python

import warnings

__version__ = "1.10.4"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
