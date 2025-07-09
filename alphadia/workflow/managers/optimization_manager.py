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
            self.ms1_error = config["search_initial"][
                "initial_ms1_tolerance"
            ]  # TODO: rename to ms1_tolerance?
            self.ms2_error = config["search_initial"]["initial_ms2_tolerance"]

            initial_rt_tolerance = config["search_initial"]["initial_rt_tolerance"]
            self.rt_error = (
                initial_rt_tolerance
                if initial_rt_tolerance > 1
                else initial_rt_tolerance * gradient_length
            )
            self.mobility_error = config["search_initial"]["initial_mobility_tolerance"]

            self.num_candidates = config["search_initial"]["initial_num_candidates"]

            self.fwhm_rt = config["optimization_manager"]["fwhm_rt"]
            self.fwhm_mobility = config["optimization_manager"]["fwhm_mobility"]
            self.score_cutoff = config["optimization_manager"]["score_cutoff"]

            self.column_type = "library"
            self.classifier_version = -1

            for key in [
                "ms1_error",
                "ms2_error",
                "rt_error",
                "mobility_error",
                "num_candidates",
                "fwhm_rt",
                "fwhm_mobility",
                "score_cutoff",
                "column_type",
                "classifier_version",
            ]:
                self.reporter.log_string(
                    f"initial parameter: {key} = {self.__dict__[key]}"
                )

    def fit(self, update_dict):  # TODO make this interface explicit
        """Update the parameters dict with the values in update_dict."""
        self.__dict__.update(update_dict)
