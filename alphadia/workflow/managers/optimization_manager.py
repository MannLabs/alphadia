import logging

from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


class OptimizationManager(BaseManager):
    ms1_error: float
    ms2_error: float
    rt_error: float
    mobility_error: float
    column_type: str
    num_candidates: int
    classifier_version: int
    fwhm_rt: float
    fwhm_mobility: float
    score_cutoff: float

    def __init__(
        self,
        config: None | Config = None,
        gradient_length: None | float = None,
        path: None | str = None,
        load_from_file: bool = True,
        **kwargs,
    ):
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)
        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            rt_error = (
                config["search_initial"]["initial_rt_tolerance"]
                if config["search_initial"]["initial_rt_tolerance"] > 1
                else config["search_initial"]["initial_rt_tolerance"] * gradient_length
            )
            initial_parameters = {
                "ms1_error": config["search_initial"]["initial_ms1_tolerance"],
                "ms2_error": config["search_initial"]["initial_ms2_tolerance"],
                "rt_error": rt_error,
                "mobility_error": config["search_initial"][
                    "initial_mobility_tolerance"
                ],
                "column_type": "library",
                "num_candidates": config["search_initial"]["initial_num_candidates"],
                "classifier_version": -1,
                "fwhm_rt": config["optimization_manager"]["fwhm_rt"],
                "fwhm_mobility": config["optimization_manager"]["fwhm_mobility"],
                "score_cutoff": config["optimization_manager"]["score_cutoff"],
            }
            self.__dict__.update(
                initial_parameters
            )  # TODO either store this as a dict or in individual instance variables

            for key, value in initial_parameters.items():
                self.reporter.log_string(f"initial parameter: {key} = {value}")

    def fit(
        self, update_dict
    ):  # TODO siblings' implementations have different signatures
        """Update the parameters dict with the values in update_dict."""
        self.__dict__.update(update_dict)
        self.is_fitted = True

    def predict(self):
        """Return the parameters dict."""
        return self.parameters

    def fit_predict(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict."""
        self.fit(update_dict)
        return self.predict()
