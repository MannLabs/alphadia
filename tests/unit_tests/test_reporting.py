import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from conftest import random_tempfolder
from matplotlib import pyplot as plt

from alphadia.constants.settings import FIGURES_FOLDER_NAME
from alphadia.reporting import reporting
from alphadia.reporting.reporting import move_existing_file


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_logging():
    reporting.__is_initiated__ = False

    tempfolder = random_tempfolder()

    if os.path.exists(os.path.join(tempfolder, "log.txt")):
        os.remove(os.path.join(tempfolder, "log.txt"))

    reporting.init_logging(tempfolder)

    python_logger = logging.getLogger()
    python_logger.progress("test")
    python_logger.info("test")
    python_logger.warning("test")
    python_logger.error("test")
    python_logger.critical("test")

    assert os.path.exists(os.path.join(tempfolder, "log.txt"))
    with open(os.path.join(tempfolder, "log.txt")) as f:
        assert len(f.readlines()) == 5
    time.sleep(1)
    os.remove(os.path.join(tempfolder, "log.txt"))
    time.sleep(1)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_backend():
    reporting.__is_initiated__ = False

    backend = reporting.Backend()
    backend.log_event("start_extraction", None)
    backend.log_metric("accuracy", 0.9)
    backend.log_string("test")
    backend.log_figure("scatter", None)
    backend.log_data("test", None)


test_backend()


def test_figure_backend():
    reporting.__is_initiated__ = False

    tempfolder = random_tempfolder()

    figure_backend = reporting.FigureBackend(path=tempfolder)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(np.random.rand(10), np.random.rand(10))

    figure_backend.log_figure("scatter", fig)
    plt.close(fig)

    assert os.path.exists(os.path.join(tempfolder, FIGURES_FOLDER_NAME, "scatter.png"))
    os.remove(os.path.join(tempfolder, FIGURES_FOLDER_NAME, "scatter.png"))
    time.sleep(1)


test_figure_backend()


def test_jsonl_backend():
    reporting.__is_initiated__ = False

    tempfolder = random_tempfolder()

    with reporting.JSONLBackend(path=tempfolder) as jsonl_backend:
        jsonl_backend.log_event("start_extraction", None)
        jsonl_backend.log_metric("accuracy", 0.9)
        jsonl_backend.log_string("test")

    assert os.path.exists(os.path.join(tempfolder, "events.jsonl"))
    with open(os.path.join(tempfolder, "events.jsonl")) as f:
        assert len(f.readlines()) == 5
    time.sleep(1)
    os.remove(os.path.join(tempfolder, "events.jsonl"))
    time.sleep(1)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_log_backend():
    reporting.__is_initiated__ = False

    tempfolder = random_tempfolder()

    if os.path.exists(os.path.join(tempfolder, "log.txt")):
        os.remove(os.path.join(tempfolder, "log.txt"))

    stdout_backend = reporting.LogBackend(path=tempfolder)
    stdout_backend.log_string("test", verbosity="progress")
    stdout_backend.log_string("test", verbosity="info")
    stdout_backend.log_string("test", verbosity="warning")
    stdout_backend.log_string("test", verbosity="error")
    stdout_backend.log_string("test", verbosity="critical")

    assert os.path.exists(os.path.join(tempfolder, "log.txt"))
    with open(os.path.join(tempfolder, "log.txt")) as f:
        assert len(f.readlines()) == 5
    # time.sleep(1)
    os.remove(os.path.join(tempfolder, "log.txt"))


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_pipeline():
    reporting.__is_initiated__ = False

    tempfolder = random_tempfolder()

    pipeline = reporting.Pipeline(
        backends=[
            reporting.LogBackend(path=tempfolder),
            reporting.JSONLBackend(path=tempfolder),
            reporting.FigureBackend(path=tempfolder),
        ]
    )

    with pipeline.context:
        pipeline.log_event("start_extraction", None)
        pipeline.log_metric("accuracy", 0.9)
        pipeline.log_string("test")

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(np.random.rand(10), np.random.rand(10))
        pipeline.log_figure("scatter", fig)
        plt.close(fig)

    assert os.path.exists(os.path.join(tempfolder, "log.txt"))
    assert os.path.exists(os.path.join(tempfolder, "events.jsonl"))
    assert os.path.exists(os.path.join(tempfolder, "figures", "scatter.png"))

    os.remove(os.path.join(tempfolder, "log.txt"))
    os.remove(os.path.join(tempfolder, "events.jsonl"))
    os.remove(os.path.join(tempfolder, "figures", "scatter.png"))

    # sleep 1 second to ensure that the file has been deleted
    time.sleep(1)


@patch("alphadia.workflow.reporting.Path.rename")
@patch("alphadia.workflow.reporting.Path.exists")
def test_move_existing_file_increments_log_file_name_and_moves(
    mock_exists, mock_rename
):
    """Test that the old log file is moved to log.1.txt"""
    mock_exists.side_effect = [True, True, False]
    log_file_path = "log.txt"
    # when
    result = move_existing_file(log_file_path)
    mock_rename.assert_called_once_with(Path("log.1.bkp.txt"))
    assert result == "log.1.bkp.txt"


@patch("alphadia.workflow.reporting.Path.rename")
@patch("alphadia.workflow.reporting.Path.exists")
def test_move_existing_file_returns_none_when_no_existing_old_log(
    mock_exists, mock_rename
):
    """Test that the function returns None when there is no existing old log file"""
    mock_exists.return_value = False
    log_file_path = "log.txt"
    # when
    result = move_existing_file(log_file_path)
    mock_rename.assert_not_called()
    assert result is None
