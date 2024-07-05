# native imports
import base64
import json
import logging
import os
import time
import traceback
import typing
import warnings
from datetime import datetime, timedelta
from io import BytesIO

# alphadia imports
# alpha family imports
# third party imports
import matplotlib
import numpy as np
from matplotlib.figure import Figure

# global variable which tracks if any logger has been initiated
# As soon as its instantiated the default logger will be configured with a path to save the log file
__is_initiated__ = False

# Add a new logging level to the default logger, level 21 is just above INFO (20)
# This has to happen at load time to make the .progress() method available even if no logger is instantiated
PROGRESS_LEVELV_NUM = 21
logging.PROGRESS = PROGRESS_LEVELV_NUM
logging.addLevelName(PROGRESS_LEVELV_NUM, "PROGRESS")


def progress(self, message, *args, **kws):
    if self.isEnabledFor(PROGRESS_LEVELV_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(PROGRESS_LEVELV_NUM, message, args, **kws)


logging.Logger.progress = progress


class DefaultFormatter(logging.Formatter):
    template = "%(levelname)s: %(message)s"

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    def __init__(self, use_ansi: bool = True):
        """
        Default formatter adding elapsed time and optional ANSI colors.

        Parameters
        ----------

        use_ansi : bool, default True
            Whether to use ANSI escape codes to color the output.

        """
        self.start_time = time.time()

        if use_ansi:
            self.formatter = {
                logging.DEBUG: logging.Formatter(self.template),
                logging.INFO: logging.Formatter(self.template),
                logging.PROGRESS: logging.Formatter(
                    self.green + self.template + self.reset
                ),
                logging.WARNING: logging.Formatter(
                    self.yellow + self.template + self.reset
                ),
                logging.ERROR: logging.Formatter(self.red + self.template + self.reset),
                logging.CRITICAL: logging.Formatter(
                    self.bold_red + self.template + self.reset
                ),
            }
        else:
            self.formatter = {
                logging.DEBUG: logging.Formatter(self.template),
                logging.INFO: logging.Formatter(self.template),
                logging.PROGRESS: logging.Formatter(self.template),
                logging.WARNING: logging.Formatter(self.template),
                logging.ERROR: logging.Formatter(self.template),
                logging.CRITICAL: logging.Formatter(self.template),
            }

    def format(self, record: logging.LogRecord):
        """Format the log record.

        Parameters
        ----------

        record : logging.LogRecord
            Log record to format.

        Returns
        -------
        str
            Formatted log record.
        """

        elapsed_seconds = record.created - self.start_time
        elapsed = timedelta(seconds=elapsed_seconds)

        return f"{elapsed} {self.formatter[record.levelno].format(record)}"


def init_logging(
    log_folder: str = None, log_level: int = logging.INFO, overwrite: bool = True
):
    """Initialize the default logger.
    Sets the formatter and the console and file handlers.

    Parameters
    ----------

    log_folder : str, default None
        Path to the folder where the log file will be saved. If None, the log file will not be saved.

    log_level : int, default logging.INFO
        Log level to use. Can be logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR or logging.CRITICAL.

    overwrite : bool, default True
        Whether to overwrite the log file if it already exists.
    """

    global __is_initiated__

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(log_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(DefaultFormatter(use_ansi=True))

    # add the handlers to the logger
    logger.addHandler(ch)

    if log_folder is not None:
        log_name = os.path.join(log_folder, "log.txt")
        # check if log file exists
        if os.path.exists(log_name) and overwrite:
            # if it does, delete it
            os.remove(log_name)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_name, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(DefaultFormatter(use_ansi=False))
        logger.addHandler(fh)

    __is_initiated__ = True


class Backend:
    """Generic backend for logging metrics, plots and strings.

    Implementations of the backend can implement the `log_figure`, `log_metric`, `log_string` and `log_data` methods.
    The subclasses implement their own logic for handling the data and writing it to a file or a database.

    If the backend requires a context, it can implement the `__enter__` and `__exit__` methods.
    The parent `MetricLogger` class will call these methods when entering and exiting the context of a single workflow.
    """

    REQUIRES_CONTEXT = False

    def log_figure(self, name: str, figure: typing.Any, *args, **kwargs):
        pass

    def log_metric(self, name: str, value: float, *args, **kwargs):
        pass

    def log_string(self, value: str, *args, **kwargs):
        pass

    def log_data(self, name: str, value: typing.Any, *args, **kwargs):
        pass

    def log_event(self, name: str, value: typing.Any, *args, **kwargs):
        pass


class FigureBackend(Backend):
    FIGURE_PATH = "figures"

    def __init__(self, path=None, default_savefig_kwargs=None) -> None:
        """Backend which logs figures to a folder.

        implements the `log_figure` method.

        Parameters
        ----------

        path : str, default None
            Path to the parent folder where the figures will be saved. If None, an error will be raised.
            The figures will be saved in a subfolder named after the class name of the backend.

        default_savefig_kwargs : dict, default {"dpi":300}
            Default arguments to pass to matplotlib.figure.Figure.savefig

        """
        if default_savefig_kwargs is None:
            default_savefig_kwargs = {"dpi": 300}
        self.path = path

        if self.path is None:
            raise ValueError(
                "FigureBackend requires an output folder to be set with the path parameter."
            )

        # create figures folder if it does not exist
        self.figures_path = os.path.join(self.path, self.FIGURE_PATH)
        os.makedirs(self.figures_path, exist_ok=True)

        self.default_savefig_kwargs = default_savefig_kwargs

    def log_figure(
        self,
        name: str,
        figure: Figure | np.ndarray,
        extension: str = "png",
    ):
        """Log a figure to the figures folder.

        Parameters
        ----------

        name : str
            Name of the figure. Will be used as the filename.

        figure : typing.Union[matplotlib.figure.Figure, np.ndarray]
            Figure to log. Can be a matplotlib figure or a numpy array.

        extension : str, default 'png'
            Extension to use for the figure. Can be any extension supported by matplotlib.image.imsave

        """

        filename = os.path.join(self.figures_path, f"{name}.{extension}")

        if isinstance(figure, matplotlib.figure.Figure):
            figure.savefig(filename, **self.default_savefig_kwargs)
        elif isinstance(figure, np.ndarray):
            matplotlib.image.imsave(filename, figure, **self.default_savefig_kwargs)
        else:
            warnings.warn(f"FigureBackend does not support type {type(figure)}")


class JSONLBackend(Backend):
    EVENTS_PATH = "events.jsonl"
    REQUIRES_CONTEXT = True

    def __init__(
        self,
        path=None,
        enable_figure=True,
        default_savefig_kwargs=None,
    ) -> None:
        """Backend which logs metrics, plots and strings to a JSONL file.
        It implements `log_figure`, `log_metric` , `log_string` and `log_event` methods.

        Important: This backend requires a context to be used.

        Parameters
        ----------

        path : str, default None
            Path to the parent folder where the output will be saved as `events.jsonl`. If None, an error will be raised.

        default_savefig_kwargs : dict, default {"dpi":300}
            Default arguments to pass to matplotlib.figure.Figure.savefig

        enable_figure : bool, default True
            Whether to enable logging of figures. If False, the `log_figure` method will be a no-op.

        """

        if default_savefig_kwargs is None:
            default_savefig_kwargs = {"dpi": 300}
        self.path = path

        if self.path is None:
            raise ValueError(
                "JSONLBackend requires an output folder to be set with the path parameter."
            )

        self.events_path = os.path.join(self.path, self.EVENTS_PATH)
        self.default_savefig_kwargs = default_savefig_kwargs
        self.enable_figure = enable_figure
        self.entered_context = False
        self.start_time = 0

    def absolute_time(self):
        """Get the current time as an ISO 8601 string.

        Returns
        -------

        str
            Current time as an ISO 8601 string.
        """
        return datetime.now().isoformat()

    def relative_time(self):
        """Get the time since the context was entered in seconds.

        Returns
        -------

        str
            Time since the context was entered.

        """
        return datetime.now().timestamp() - self.start_time

    def __enter__(self):
        """Enter the context of the backend.
        This method will create an empty `events.jsonl` file and write a `start` event to it.
        """

        self.entered_context = True
        self.start_time = datetime.now().timestamp()

        # empty the file if it exists
        with open(self.events_path, "w"):
            pass

        self.log_event("start", {})
        return self

    def __exit__(
        self, exc_type: typing.Any, exc_value: typing.Any, exc_traceback: typing.Any
    ):
        """Exit the context of the backend.
        This method will write a `stop` event to the `events.jsonl` file.

        Parameters
        ----------

        exc_type : typing.Any
            Type of the exception raised. If no exception was raised, this will be None.

        exc_value : typing.Any
            Value of the exception raised. If no exception was raised, this will be None.

        exc_traceback : typing.Any
            Traceback of the exception raised. If no exception was raised, this will be None.
        """

        if exc_type is not None:
            exc_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )

            self.log_event("stop", {"error": exc_str})
        else:
            self.log_event("stop", {})

        self.entered_context = False
        self.start_time = 0

    def log_event(self, name: str, value: typing.Any):
        """Log an event to the `events.jsonl` file.

        Important: This method will only log events if the backend is in a context.
        Otherwise, it will be a no-op.

        Parameters
        ----------

        name : str
            Name of the event.

        value : typing.Any
            Value of the event. Must be a JSON-serializable object.
        """

        if not self.entered_context:
            return

        with open(self.events_path, "a") as f:
            message = {
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "event",
                "name": name,
                "value": value,
                "verbosity": 0,
            }
            f.write(json.dumps(message) + "\n")

    def log_metric(self, name: str, value: float):
        """Log a metric to the `events.jsonl` file.

        Important: This method will only log metrics if the backend is in a context.
        Otherwise, it will be a no-op.

        Parameters
        ----------

        name : str
            Name of the metric.

        value : float
            Value of the metric.
        """

        if not self.entered_context:
            return

        with open(self.events_path, "a") as f:
            message = {
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "metric",
                "name": name,
                "value": value,
                "verbosity": 0,
            }
            f.write(json.dumps(message) + "\n")

    def log_string(self, value: str, verbosity: int = "info"):
        """Log a string to the `events.jsonl` file.

        Important: This method will only log strings if the backend is in a context.
        Otherwise, it will be a no-op.

        Parameters
        ----------

        value : str
            Value of the string.

        verbosity : int, default 0
            Verbosity of the string. Can later be used to filter strings.

        """
        if not self.entered_context:
            return

        with open(self.events_path, "a") as f:
            message = {
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "string",
                "name": "string",
                "value": value,
                "verbosity": verbosity,
            }
            f.write(json.dumps(message) + "\n")

    def log_figure(self, name: str, figure: typing.Any):
        """Log a base64 image of a figure to the `events.jsonl` file.

        Important: This method will only log figures if the backend is in a context.
        Otherwise, it will be a no-op.

        Parameters
        ----------

        name : str
            Name of the figure.

        figure : typing.Any
            Figure to log. Can be a matplotlib figure or a numpy array.

        """

        if not self.entered_context or not self.enable_figure:
            return

        buffer = BytesIO()
        if isinstance(figure, matplotlib.figure.Figure):
            figure.savefig(buffer, **self.default_savefig_kwargs)
        elif isinstance(figure, np.ndarray):
            matplotlib.image.imsave(buffer, figure, **self.default_savefig_kwargs)
        else:
            warnings.warn(f"FigureBackend does not support type {type(figure)}")
            return

        base64_encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        with open(self.events_path, "a") as f:
            message = {
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "figure",
                "name": name,
                "value": base64_encoded_image,
                "verbosity": 0,
            }
            f.write(json.dumps(message) + "\n")


class NeptuneBackend(Backend):
    pass


class LogBackend(Backend):
    def __init__(self, path: str = None) -> None:
        if not __is_initiated__ or path is not None:
            init_logging(path)

        self.logger = logging.getLogger()
        super().__init__()

    def log_string(self, value: str, verbosity: str = "info"):
        if verbosity == "progress":
            self.logger.progress(value)
        elif verbosity == "info":
            self.logger.info(value)
        elif verbosity == "debug":
            self.logger.debug(value)
        elif verbosity == "warning":
            self.logger.warning(value)
        elif verbosity == "error":
            self.logger.error(value)
        elif verbosity == "critical":
            self.logger.critical(value)
        else:
            raise ValueError(f"Unknown verbosity level {verbosity}")


class Context:
    def __init__(self, parent: typing.Any) -> None:
        """Helper class to allow backends to use a context manager.
        This allows the parent class to be instantiated without context and to receive  context later.

        Parameters
        ----------

        parent : typing.Any
            The metric logger which owns this context

        """
        self.parent = parent

    def __enter__(self):
        return self.parent.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self.parent.__exit__(exc_type, exc_value, exc_traceback)


class Pipeline:
    def __init__(
        self,
        backends: list[type[Backend]] = None,
    ):
        """Metric logger which allows to log metrics, plots and strings to multiple backends.

        Parameters
        ----------

        backends : typing.List[Type[Backend]], default [LogBackend]
            typing.List of backends to use. Each backend must be a class inheriting from Backend.
        """

        # the context will store a Context object
        # this allows backends which require a context to be used
        if backends is None:
            backends = []
        self.context = Context(self)

        # instantiate backends
        self.backends = backends

    def __enter__(self):
        for backend in self.backends:
            if backend.REQUIRES_CONTEXT:
                backend.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for backend in self.backends:
            if backend.REQUIRES_CONTEXT:
                backend.__exit__(exc_type, exc_value, exc_traceback)

    def log_figure(self, name: str, figure: typing.Any, *args, **kwargs):
        for backend in self.backends:
            backend.log_figure(name, figure, *args, **kwargs)

    def log_metric(self, name: str, value: float, *args, **kwargs):
        for backend in self.backends:
            backend.log_metric(name, value, *args, **kwargs)

    def log_string(self, value: str, *args, verbosity="info", **kwargs):
        for backend in self.backends:
            backend.log_string(value, *args, verbosity=verbosity, **kwargs)

    def log_data(self, name: str, value: typing.Any, *args, **kwargs):
        for backend in self.backends:
            backend.log_data(name, value, *args, **kwargs)

    def log_event(self, name: str, value: typing.Any, *args, **kwargs):
        for backend in self.backends:
            backend.log_event(name, value, *args, **kwargs)
