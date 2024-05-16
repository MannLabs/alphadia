"""Calculate metrics and upload them to neptune.

To extend the metrics, create a new class that inherits from Metrics and implement the _calc() method.
"""

import os
import sys
from abc import ABC
from typing import Union

import pandas as pd
import neptune

from tests.e2e_tests.prepare_test_data import get_test_case, OUTPUT_DIR_NAME

NEPTUNE_PROJECT_NAME = os.environ.get("NEPTUNE_PROJECT_NAME")


def _load_tsv(file_path: str) -> pd.DataFrame:
    """Load a tsv file."""
    return pd.read_csv(file_path, sep="\t")


def _load_hdf(file_path: str) -> dict[str, pd.DataFrame]:
    """Load a hdf file into a dictionary of keys."""
    hdfs = {}
    with pd.HDFStore(file_path) as store:
        for key in store.keys():
            hdfs[key] = pd.read_hdf(file_path, key)
    return hdfs


file_name_to_read_method_mapping = {
    "pg.matrix.tsv": _load_tsv,
    "precursors.tsv": _load_tsv,
    "stat.tsv": _load_tsv,
    "speclib.hdf": _load_hdf,
    "speclib.mbr.hdf": _load_hdf,
}


class DataStore:
    def __init__(self, data_dir: str):
        """Data store to cache data."""
        self._data_dir = data_dir + "/"
        self._data = {}

    def __getitem__(self, key: str) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Get data from the data store."""
        if key not in self._data:
            file_path = self._data_dir + key
            print("loading", file_path)
            self._data[key] = file_name_to_read_method_mapping[key](file_path)
        return self._data[key]


class Metrics(ABC):
    def __init__(self, data_store: DataStore):
        """Abstract class for metrics."""
        self._data_store = data_store
        self._metrics = {}
        self._name = self.__class__.__name__

    def get(self):
        """Get the metrics."""
        if not self._metrics:
            self._calc()
        return self._metrics

    def _calc(self):
        """Calculate the metrics."""
        raise NotImplementedError


class BasicStats(Metrics):
    """Basic statistics."""

    def _calc(self):
        """Calculate metrics."""
        df = self._data_store["stat.tsv"]

        for col in ["proteins", "precursors", "ms1_accuracy", "fwhm_rt"]:
            self._metrics[f"{self._name}/{col}_mean"] = df[col].mean()
            self._metrics[f"{self._name}/{col}_std"] = df[col].std()


if __name__ == "__main__":
    test_case_name = sys.argv[1]

    test_case = get_test_case(test_case_name)
    selected_metrics = test_case["metrics"]  # ['BasicStats', "BasicStats2"]

    run = {}
    run["test_case"] = test_case_name
    # neptune_run["config"] = # TODO add config ?
    # neptune_run["commit_hash"] = # TODO add more metadata: commit hash, ...

    try:
        data_store = DataStore(test_case_name + "/" + OUTPUT_DIR_NAME)

        metrics_classes = [
            cls for cls in Metrics.__subclasses__() if cls.__name__ in selected_metrics
        ]

        for cl in metrics_classes:
            metrics = cl(data_store).get()
            print(cl, metrics)
            run |= metrics
    except Exception as e:
        print(e)
    finally:
        neptune_run = neptune.init_run(
            project=NEPTUNE_PROJECT_NAME, tags=test_case_name
        )
        for k, v in run.items():
            neptune_run[k] = v
        neptune_run.stop()
