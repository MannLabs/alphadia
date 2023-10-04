from matplotlib import pyplot as plt
import numpy as np
import tempfile
import os
import logging
import time

from alphadia.extraction.workflow import reporting

def test_logging():
    
    tempfolder = tempfile.gettempdir()

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
    with open(os.path.join(tempfolder, "log.txt"), "r") as f:
        assert len(f.readlines()) == 5
    time.sleep(1)
    os.remove(os.path.join(tempfolder, "log.txt"))
    time.sleep(1)

test_logging()

def test_backend():

    backend = reporting.Backend()
    backend.log_event("start_extraction", None)
    backend.log_metric("accuracy", 0.9)
    backend.log_string("test")
    backend.log_figure("scatter", None)
    backend.log_data("test", None)

test_backend()

def test_figure_backend():

    figure_backend = reporting.FigureBackend(path = tempfile.gettempdir())

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(np.random.rand(10), np.random.rand(10))

    figure_backend.log_figure("scatter", fig)
    plt.close(fig)

    assert os.path.exists(os.path.join(tempfile.gettempdir(), figure_backend.FIGURE_PATH, "scatter.png"))
    os.remove(os.path.join(tempfile.gettempdir(), figure_backend.FIGURE_PATH, "scatter.png"))
    time.sleep(1)

test_figure_backend()

def test_jsonl_backend():
    
    with reporting.JSONLBackend(path = tempfile.gettempdir()) as jsonl_backend:
        jsonl_backend.log_event("start_extraction", None)
        jsonl_backend.log_metric("accuracy", 0.9)
        jsonl_backend.log_string("test")

    assert os.path.exists(os.path.join(tempfile.gettempdir(), "events.jsonl"))
    with open(os.path.join(tempfile.gettempdir(), "events.jsonl"), "r") as f:
        assert len(f.readlines()) == 5
    time.sleep(1)
    os.remove(os.path.join(tempfile.gettempdir(), "events.jsonl"))
    time.sleep(1)

test_jsonl_backend()

def test_log_backend():

    tempdir = tempfile.gettempdir()

    if os.path.exists(os.path.join(tempdir, "log.txt")):
        os.remove(os.path.join(tempdir, "log.txt"))

    stdout_backend = reporting.LogBackend(path = tempdir)
    stdout_backend.log_string("test", verbosity='progress')
    stdout_backend.log_string("test", verbosity='info')
    stdout_backend.log_string("test", verbosity='warning')
    stdout_backend.log_string("test", verbosity='error')
    stdout_backend.log_string("test", verbosity='critical')

    assert os.path.exists(os.path.join(tempdir, "log.txt"))
    with open(os.path.join(tempdir, "log.txt"), "r") as f:
        assert len(f.readlines()) == 5
    time.sleep(1)
    os.remove(os.path.join(tempdir, "log.txt"))

test_log_backend()

def test_pipeline():

    tempdir = tempfile.gettempdir()

    pipeline = reporting.Pipeline(
        backends = [
            reporting.LogBackend(path = tempdir),
            reporting.JSONLBackend(path = tempdir),
            reporting.FigureBackend(path = tempdir)
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

    assert os.path.exists(os.path.join(tempdir, "log.txt"))
    assert os.path.exists(os.path.join(tempdir, "events.jsonl"))
    assert os.path.exists(os.path.join(tempdir, "figures", "scatter.png"))

    os.remove(os.path.join(tempdir, "log.txt"))
    os.remove(os.path.join(tempdir, "events.jsonl"))
    os.remove(os.path.join(tempdir, "figures", "scatter.png"))

    # sleep 1 second to ensure that the file has been deleted
    time.sleep(1)

test_pipeline()