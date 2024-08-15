"""Calculate metrics and upload them to neptune.

To extend the metrics, create a new class that inherits from Metrics and implement the _calc() method.
"""

import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import neptune
import pandas as pd

from tests.e2e_tests.prepare_test_data import OUTPUT_DIR_NAME, get_test_case

NEPTUNE_PROJECT_NAME = os.environ.get("NEPTUNE_PROJECT_NAME")


def _load_tsv(file_path: str) -> pd.DataFrame:
    """Load a tsv file."""
    return pd.read_csv(file_path, sep="\t")


def _load_speclib_hdf(file_path: str) -> dict[str, pd.DataFrame]:
    """Load a hdf file into a dictionary of keys."""
    raise NotImplementedError("Loading speclib hdf is not implemented yet.")


class OutputFiles:
    """String constants for the output file names."""

    PG_MATRIX = "pg.matrix.tsv"
    PRECURSORS = "precursors.tsv"
    STAT = "stat.tsv"
    LOG = "log.txt"
    # SPECLIB = "speclib.hdf"
    # SPECLIB_MBR = "speclib.mbr.hdf"

    @classmethod
    def all_values(cls) -> list[str]:
        """Get all values of the class as a list of str."""
        return [
            v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and k != "all_values"
        ]


file_name_to_read_method_mapping = {
    OutputFiles.PG_MATRIX: _load_tsv,
    OutputFiles.PRECURSORS: _load_tsv,
    OutputFiles.STAT: _load_tsv,
    # OutputFiles.SPECLIB: _load_speclib_hdf,
    # OutputFiles.SPECLIB_MBR: _load_speclib_hdf,
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

    @abstractmethod
    def _calc(self) -> None:
        """Calculate the metrics."""
        raise NotImplementedError


class BasicStats(Metrics):
    """Basic statistics."""

    def _calc(self):
        """Calculate metrics."""
        df = self._data_store[OutputFiles.STAT]

        for col in ["proteins", "precursors", "ms1_accuracy", "fwhm_rt"]:
            self._metrics[f"{self._name}/{col}_mean"] = df[col].mean()
            self._metrics[f"{self._name}/{col}_std"] = df[col].std()


def _basic_plot(df: pd.DataFrame, test_case: str, metric: str, metric_std: str = None):
    """Draw a basic line plot of `metric` for `test_case` over time."""

    df = (
        df[df["test_case"] == test_case]
        .sort_index(ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots()
    ax.scatter(x=df.index, y=df[metric])
    if metric_std:
        ax.errorbar(x=df.index, y=df[metric], yerr=df[metric_std])

    ax.set_title(f"test_case: {test_case}, metric: {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("test runs")

    labels = []
    for x, y, z in zip(
        df["sys/creation_time"],
        df["branch_name"],
        df["short_sha"],
        strict=True,
    ):
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        dt = datetime.strptime(str(x), fmt)
        x = dt.strftime("%Y%m%d_%H:%M:%S")

        labels.append(f"{x}:\n{y} [{z}]")

    ax.set_xticks(df.index, labels, rotation=66)

    return fig


def _get_history_plots(test_results: dict, metrics_classes: list):
    """Get all past runs from neptune, add the current one and create plots."""

    test_results = test_results.copy()
    test_results["sys/creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    test_results_df = pd.DataFrame(test_results, index=[0])

    project = neptune.init_project(project=NEPTUNE_PROJECT_NAME, mode="read-only")
    runs_table_df = project.fetch_runs_table().to_pandas()

    df = pd.concat([runs_table_df, test_results_df])

    test_case_name = test_results["test_case"]

    figs = []
    for metrics_class in [cls.__name__ for cls in metrics_classes]:
        # TODO find a smarter way to get the metrics
        for metric in [k for k in test_results if k.startswith(metrics_class)]:
            fig = _basic_plot(df, test_case_name, metric)
            figs.append((metric, fig))

    return figs


if __name__ == "__main__":
    test_case_name = sys.argv[1]
    run_time_minutes = int(sys.argv[2]) / 60
    neptune_upload = sys.argv[3] == "True"
    short_sha = sys.argv[4]
    branch_name = sys.argv[5]

    test_case = get_test_case(test_case_name)
    selected_metrics = test_case["metrics"]  # ['BasicStats', ]

    test_results = {}
    test_results["test_case"] = test_case_name
    test_results["run_time_minutes"] = run_time_minutes
    test_results["short_sha"] = short_sha
    test_results["branch_name"] = branch_name
    test_results["test_case_details"] = str(test_case)

    output_path = os.path.join(test_case_name, OUTPUT_DIR_NAME)

    metrics_classes = [
        cls for cls in Metrics.__subclasses__() if cls.__name__ in selected_metrics
    ]

    try:
        data_store = DataStore(output_path)

        for cl in metrics_classes:
            metrics = cl(data_store).get()
            print(cl, metrics)
            test_results |= metrics
    except Exception as e:
        print(e)

    print(test_results)

    if not neptune_upload:
        print("skipping neptune upload")
        exit(0)

    neptune_run = neptune.init_run(
        project=NEPTUNE_PROJECT_NAME,
        tags=[test_case_name, short_sha, branch_name],
    )

    # metrics
    for k, v in test_results.items():
        print(f"adding {k}={v}")
        neptune_run[k] = v

    # files
    for file_name in OutputFiles.all_values():
        print("adding", file_name)
        file_path = os.path.join(output_path, file_name)
        if os.path.exists(file_path):
            neptune_run["output/" + file_name].track_files(file_path)

    try:
        history_plots = _get_history_plots(test_results, metrics_classes)

        for name, plot in history_plots:
            neptune_run[f"plots/{name}"].upload(plot)
    except Exception as e:
        print(f"no plots today: {e}")

    neptune_run.stop()
