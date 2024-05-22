"""Calculate metrics and upload them to neptune.

To extend the metrics, create a new class that inherits from Metrics and implement the _calc() method.
"""

import os
import sys
from abc import ABC
from typing import Any

import pandas as pd
import neptune

from tests.e2e_tests.prepare_test_data import get_test_case, OUTPUT_DIR_NAME

NEPTUNE_PROJECT_NAME = os.environ.get("NEPTUNE_PROJECT_NAME")


def _load_tsv(file_path: str) -> pd.DataFrame:
    """Load a tsv file."""
    return pd.read_csv(file_path, sep="\t")


def _load_speclib_hdf(file_path: str) -> dict[str, pd.DataFrame]:
    """Load a hdf file into a dictionary of keys."""
    raise NotImplementedError("Loading speclib hdf is not implemented yet.")


file_name_to_read_method_mapping = {
    "pg.matrix.tsv": _load_tsv,
    "precursors.tsv": _load_tsv,
    "stat.tsv": _load_tsv,
    "speclib.hdf": _load_speclib_hdf,
    "speclib.mbr.hdf": _load_speclib_hdf,
}


class DataStore:
    def __init__(self, data_dir: str):
        """Data store to read and cache data.

        Output files defined in `file_name_to_read_method_mapping` can be accessed as attributes, e.g.
            `stat_df = DataStore('/home/output')["stat.tsv"]`

        Parameters
        ----------
        data_dir : str
            Absolute path to the directory containing alphaDIA output data.
        """
        self._data_dir = data_dir
        self._data = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Get data from the data store."""
        if key not in self._data:
            file_path = os.path.join(self._data_dir, key)
            print("loading", file_path)
            self._data[key] = file_name_to_read_method_mapping[key](file_path)
        return self._data[key]


class Metrics(ABC):
    """Abstract class for metrics."""

    def __init__(self, data_store: DataStore):
        """Initialize Metrics.

        Parameters
        ----------
        data_store : DataStore
            Data store to get the data from.
        """

        self._data_store = data_store
        self._metrics = {}
        self._name = self.__class__.__name__

    def get(self) -> dict[str, Any]:
        """Get the metrics."""
        if not self._metrics:
            self._calc()
        return self._metrics

    def _calc(self) -> None:
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
    short_sha = sys.argv[2]
    branch_name = sys.argv[3]

    test_case = get_test_case(test_case_name)
    selected_metrics = test_case["metrics"]  # ['BasicStats', "BasicStats2"]

    test_results = {}
    test_results["test_case"] = test_case_name
    # test_results["config"] = # TODO add config ?
    # test_results["commit_hash"] = # TODO add more metadata: commit hash, ...

    try:
        data_store = DataStore(os.path.join(test_case_name, OUTPUT_DIR_NAME))

        metrics_classes = [
            cls for cls in Metrics.__subclasses__() if cls.__name__ in selected_metrics
        ]

        for cl in metrics_classes:
            metrics = cl(data_store).get()
            print(cl, metrics)
            test_results |= metrics
    except Exception as e:
        print(e)
    finally:
        neptune_run = neptune.init_run(
            project=NEPTUNE_PROJECT_NAME,
            tags=[test_case_name, short_sha, branch_name],
        )
        for k, v in test_results.items():
            neptune_run[k] = v
        neptune_run.stop()
