import logging

import pandas as pd

from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


class TimingManager(BaseManager):
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        """Contains and updates timing information for the portions of the workflow."""
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})
        if not self.is_loaded_from_file:
            self.timings = {}

    def set_start_time(self, workflow_stage: str):
        """Stores the start time of the given stage of the workflow in the timings attribute. Also saves the timing manager to disk.

        Parameters
        ----------
        workflow_stage : str
            The name under which the timing will be stored in the timings dict
        """
        self.timings.update({workflow_stage: {"start": pd.Timestamp.now()}})

    def set_end_time(self, workflow_stage: str):
        """Stores the end time of the given stage of the workflow in the timings attribute and calculates the duration. Also saves the timing manager to disk.

        Parameters
        ----------
        workflow_stage : str
            The name under which the timing will be stored in the timings dict

        """
        self.timings[workflow_stage]["end"] = pd.Timestamp.now()
        self.timings[workflow_stage]["duration"] = (
            self.timings[workflow_stage]["end"] - self.timings[workflow_stage]["start"]
        ).total_seconds() / 60
