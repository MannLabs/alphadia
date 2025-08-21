import logging

from alphadia.constants.keys import COLUMN_TYPE_LIBRARY
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

            self.column_type = COLUMN_TYPE_LIBRARY
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

    def update(
        self,
        *,
        ms1_error: float | None = None,
        ms2_error: float | None = None,
        rt_error: float | None = None,
        mobility_error: float | None = None,
        column_type: str | None = None,
        num_candidates: int | None = None,
        classifier_version: int | None = None,
        fwhm_rt: float | None = None,
        fwhm_mobility: float | None = None,
        score_cutoff: float | None = None,
    ):
        """Update the parameters dict with the values in update_dict."""
        if ms1_error is not None:
            self.ms1_error = ms1_error
        if ms2_error is not None:
            self.ms2_error = ms2_error
        if rt_error is not None:
            self.rt_error = rt_error
        if mobility_error is not None:
            self.mobility_error = mobility_error
        if column_type is not None:
            self.column_type = column_type
        if num_candidates is not None:
            self.num_candidates = num_candidates
        if classifier_version is not None:
            self.classifier_version = classifier_version
        if fwhm_rt is not None:
            self.fwhm_rt = fwhm_rt
        if fwhm_mobility is not None:
            self.fwhm_mobility = fwhm_mobility
        if score_cutoff is not None:
            self.score_cutoff = score_cutoff
