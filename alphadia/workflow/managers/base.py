"""Base class for Managers.

In AlphaDIA, a "manager" is a stateful object, and can be saved/loaded from disk. Additionally, it may offer functionality to change its state.
"""

import logging
import os
import pickle
import traceback

import alphadia
from alphadia.reporting import reporting

logger = logging.getLogger()


class BaseManager:
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        figure_path: None | str = None,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """Base class for all managers which handle parts of the workflow.

        Parameters
        ----------

        path : str, optional
            Path to the manager pickle on disk.

        load_from_file : bool, optional
            If True, the manager will be loaded from file if it exists.
        """

        self._path = path
        self.is_loaded_from_file = False
        self.figure_path = figure_path
        self._version = alphadia.__version__
        self.reporter = reporting.LogBackend() if reporter is None else reporter

        if load_from_file:
            # Note: be careful not to overwrite loaded values by initializing them in child classes after calling super().__init__()
            self.load()

    @property
    def path(self):
        """Path to the manager pickle on disk."""
        return self._path

    @property
    def is_loaded_from_file(self):
        """Check if the calibration manager was loaded from file."""
        return self._is_loaded_from_file

    @is_loaded_from_file.setter
    def is_loaded_from_file(self, value):
        self._is_loaded_from_file = value

    def save(self):
        """Save the state to pickle file."""
        if self.path is None:
            return

        try:
            with open(self.path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            self.reporter.log_string(
                f"Failed to save {self.__class__.__name__} to {self.path}: {str(e)}",
                verbosity="error",
            )

            self.reporter.log_string(
                f"Traceback: {traceback.format_exc()}", verbosity="error"
            )

    def load(self):
        """Load the state from pickle file."""
        if self.path is None:
            self.reporter.log_string(
                f"{self.__class__.__name__}: loading saved state not requested, will be initialized.",
            )
            return
        elif not os.path.exists(self.path):
            self.reporter.log_string(
                f"{self.__class__.__name__}: not found at {self.path}, will be initialized.",
                verbosity="warning",
            )
            return

        try:
            with open(self.path, "rb") as f:
                loaded_state = pickle.load(f)

                if loaded_state._version == self._version:
                    self.__dict__.update(loaded_state.__dict__)
                    self.is_loaded_from_file = True
                    self.reporter.log_string(
                        f"Loaded {self.__class__.__name__} from {self.path}"
                    )
                else:
                    self.reporter.log_string(
                        f"Version mismatch while loading {self.__class__}: {loaded_state._version} != {self._version}. Will not load.",
                        verbosity="warning",
                    )
        except Exception:
            self.reporter.log_string(
                f"Failed to load {self.__class__.__name__} from {self.path}",
                verbosity="error",
            )
