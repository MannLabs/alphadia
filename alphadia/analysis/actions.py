"""A module to analyse timsTOF DIA data."""

import logging
import collections
import abc

import alphatims.bruker
import alphadia.smoothing

class ActionDeque(collections.deque):

    def run_all_consecutive_actions(self) -> None:
        if len(self) == 0:
            logging.info("No actions in ActionDeque")
        else:
            if len(self) == 1:
                logging.info("Running 1 action in ActionDeque")
            else:
                logging.info(f"Running {len(self)} actions in ActionDeque")
            for action_to_take in self:
                action_to_take.run()


class Action(abc.ABC):

    def __init__(self, **parameters):
        self.update_parameters(**parameters)

    @property
    @abc.abstractmethod
    def default_parameters(self) -> dict:
        pass

    @property
    def parameters(self) -> dict:
        if not hasattr(self, "_parameters"):
            self._parameters = self.default_parameters
        return self._parameters

    def update_parameters(self, **parameters) -> None:
        self._parameters = self.parse_valid_parameters(**parameters)

    def parse_valid_parameters(self, **parameters) -> None:
        current_parameters = self.parameters
        for parameter_key, parameter_value in parameters.items():
            current_parameters[parameter_key] = parameter_value
        return current_parameters

    def set_output(self, output: type) -> type:
        self._output = output

    @property
    def output(self) -> type:
        if not hasattr(self, "_output"):
            raise ValueError("No output has been defined for this action")
        return self._output

    @property
    def is_completed(self) -> bool:
        return hasattr(self, "_output")

    def run(self, redo_completed: bool = False, **parameters) -> None:
        if redo_completed or not self.is_completed:
            if len(parameters) > 0:
                self.update_parameters(**parameters)
            logging.info(f"Running '{self.__class__.__name__}'")
            try:
                output = self._run()
                self.set_output(output)
            except Exception as raised_exception:
                if hasattr(self, "_output"):
                    del self._output
                raise raised_exception
        else:
            logging.info(
                f"'{self.__class__.__name__}' is already completed"
            )
        return self.output

    @property
    @abc.abstractmethod
    def runnable_function(self) -> callable:
        pass

    def _run(self) -> type:
        return self.runnable_function(**self.parameters)

    # @staticmethod
    # def create(name):
    #     if name == "import":
    #         return ImportAction()

class ImportAction(Action):

    @property
    def default_parameters(self) -> dict:
        return {
            "bruker_d_folder_name": None,
        }

    @property
    def runnable_function(self) -> callable:
        return alphatims.bruker.TimsTOF

class ConnectAction(Action):

    @property
    def default_parameters(self) -> dict:
        return {
            "scan_tolerance": 6,
            "dia_data": None,
            "multiple_frames": False,
            "ms1": True,
            "ms2": True,
        }

    @property
    def runnable_function(self) -> callable:
        # import functools
        # _func = functools.partial(
        #     alphadia.smoothing.get_connections_within_cycle,
        #     scan_max_index=self.parameters["dia_data"].scan_max_index,
        #     dia_mz_cycle=self.parameters["dia_data"].dia_mz_cycle
        # )
        def _func2(**kwargs):
            parameters = self.parameters.copy()
            dia_data = parameters.pop("dia_data")
            return alphadia.smoothing.get_connections_within_cycle(
                scan_max_index=dia_data.scan_max_index,
                dia_mz_cycle=dia_data.dia_mz_cycle,
                **parameters,
            )
        # parameters = self.parameters.copy()
        # dia_data = parameters.pop("dia_data")
        # _func = functools.partial(
        #     alphadia.smoothing.get_connections_within_cycle,
        #     scan_max_index=dia_data.scan_max_index,
        #     dia_mz_cycle=dia_data.dia_mz_cycle,
        #     **parameters,
        # )
        # result = _func()
        # def _func2(**kwargs):
        #     return result
        return _func2
