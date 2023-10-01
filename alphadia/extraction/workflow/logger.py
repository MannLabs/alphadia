
import typing
import os
from datetime import datetime
from typing import Any, List, Type, Union

import matplotlib
from matplotlib.figure import Figure
import numpy as np
import warnings
import json
import base64
from io import BytesIO

import traceback


class Backend():
    """Generic backend for logging metrics, plots and strings.

    Implementations of the backend can implement the `log_figure`, `log_metric`, `log_string` and `log_data` methods.
    The subclasses implement their own logic for handling the data and writing it to a file or a database.

    If the backend requires a context, it can implement the `__enter__` and `__exit__` methods.
    The parent `MetricLogger` class will call these methods when entering and exiting the context of a single workflow.
    """

    REQUIRES_CONTEXT = False
    
    def log_figure(self, name : str, figure : Any):
        pass

    def log_metric(self, name : str, value : float):
        pass

    def log_string(self, value : str):
        pass

    def log_data(self, name : str, value : Any):
        pass

    def log_event(self, name : str, value : Any):
        pass

class FigureBackend(Backend):

    FIGURE_PATH = "figures"

    def __init__(
            self, 
            path = None, 
            default_savefig_kwargs = {
                "dpi":300
            }) -> None:
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
        self.path = path
        
        if self.path is None:
            raise ValueError("FigureBackend requires an output folder to be set with the path parameter.")
        
        # create figures folder if it does not exist
        self.figures_path = os.path.join(self.path, self.FIGURE_PATH)
        os.makedirs(self.figures_path, exist_ok=True)       

        self.default_savefig_kwargs = default_savefig_kwargs 

    def log_figure(
            self, 
            name : str, 
            figure : Union[Figure, np.ndarray],
            extension : str = 'png'
        ):
        """Log a figure to the figures folder.

        Parameters
        ----------

        name : str
            Name of the figure. Will be used as the filename.

        figure : Union[matplotlib.figure.Figure, np.ndarray]
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
        path = None,
        enable_figure = True,
        default_savefig_kwargs = {
            "dpi":300
        },
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
        
        self.path = path

        if self.path is None:
            raise ValueError("JSONLBackend requires an output folder to be set with the path parameter.")

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
        with open(self.events_path, 'w') as f:
            pass

        self.log_event("start", {})
        return self

    def __exit__(
            self,
            exc_type: Any,
            exc_value: Any,
            exc_traceback: Any
        ):
        """Exit the context of the backend.
        This method will write a `stop` event to the `events.jsonl` file.

        Parameters
        ----------

        exc_type : Any
            Type of the exception raised. If no exception was raised, this will be None.

        exc_value : Any
            Value of the exception raised. If no exception was raised, this will be None.

        exc_traceback : Any
            Traceback of the exception raised. If no exception was raised, this will be None.
        """

        if exc_type is not None:
            exc_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            self.log_event("stop", {"error": exc_str})
        else:
            self.log_event("stop", {})

        self.entered_context = False
        self.start_time = 0
        
    def log_event(
            self, 
            name : str, 
            value : Any
        ):
        """Log an event to the `events.jsonl` file.

        Important: This method will only log events if the backend is in a context. 
        Otherwise, it will be a no-op.

        Parameters
        ----------

        name : str
            Name of the event.

        value : Any
            Value of the event. Must be a JSON-serializable object.
        """

        if not self.entered_context:
            return
        
        with open(self.events_path, 'a') as f:
            message = {
                
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "event",
                "name": name,
                "value": value,
                "verbosity": 0
            }
            f.write(json.dumps(message) + "\n")

    def log_metric(
            self, 
            name: str, 
            value: float
        ):
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
        
        with open(self.events_path, 'a') as f:
            message = {
                
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "metric",
                "name": name,
                "value": value,
                "verbosity": 0
            }
            f.write(json.dumps(message) + "\n")

    def log_string(
            self, 
            value: str, 
            verbosity: int = 0
        ):
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
        
        with open(self.events_path, 'a') as f:
            message = {
                
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "string",
                "name": "string",
                "value": value,
                "verbosity": verbosity
            }
            f.write(json.dumps(message) + "\n")

    def log_figure(
            self, 
            name: str, 
            figure: Any
        ):
        """Log a base64 image of a figure to the `events.jsonl` file.

        Important: This method will only log figures if the backend is in a context.
        Otherwise, it will be a no-op.

        Parameters
        ----------

        name : str
            Name of the figure.

        figure : Any
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
        
        base64_encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        with open(self.events_path, 'a') as f:
            message = {
                
                "absolute_time": self.absolute_time(),
                "relative_time": self.relative_time(),
                "type": "figure",
                "name": name,
                "value": base64_encoded_image,
                "verbosity": 0
            }
            f.write(json.dumps(message) + "\n")

class NeptuneBackend(Backend):
    pass

class StdoutBackend(Backend):

    def log_string(self, name : str, value : str):
        print(f"{name}: {value}")

    def log_metric(self, name : str, value : float):
        print(f"{name}: {value}")

class Context():
    
    def __init__(self, parent: Any) -> None:
        """Helper class to allow backends to use a context manager.
        This allows the parent class to be instantiated without context and to receive  context later.

        Parameters
        ----------

        parent : Any
            The metric logger which owns this context
        
        """
        self.parent = parent

    def __enter__(self):
        return self.parent.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        return self.parent.__exit__(exc_type, exc_value, exc_traceback)

class MetricLogger():

    def __init__(
            self,
            backends : List[Type[Backend]] = [StdoutBackend()],
    ):
        """Metric logger which allows to log metrics, plots and strings to multiple backends.

        Parameters
        ----------

        backends : List[Type[Backend]], default [StdoutBackend]
            List of backends to use. Each backend must be a class inheriting from Backend.
        """
        
        # the context will store a Context object
        # this allows backends which require a context to be used
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

    def log_figure(self, name : str, figure : Any, *args, **kwargs):
        for backend in self.backends:
            backend.log_figure(name, figure, *args, **kwargs)

    def log_metric(self, name : str, value : float, *args, **kwargs):
        for backend in self.backends:
            backend.log_metric(name, value, *args, **kwargs)

    def log_string(self, name : str, value : str, *args, **kwargs):
        for backend in self.backends:
            backend.log_string(name, value, *args, **kwargs)

    def log_data(self, name : str, value : Any, *args, **kwargs):
        for backend in self.backends:
            backend.log_data(name, value, *args, **kwargs)

    def log_event(self, name : str, value : Any, *args, **kwargs):
        for backend in self.backends:
            backend.log_event(name, value, *args, **kwargs)