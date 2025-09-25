import logging

from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


class OptimizationManager(BaseManager):
    ms1_error: float
    ms2_error: float
    rt_error: float
    mobility_error: float
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
        num_candidates: int | None = None,
        classifier_version: int | None = None,
        fwhm_rt: float | None = None,
        fwhm_mobility: float | None = None,
        score_cutoff: float | None = None,
    ):
        """Update the parameters dict with the values in update_dict."""

        update_logs = []

        def _add_update_log(name: str, old: float | str, new: float | str) -> None:
            """Add a log entry for an updated parameter."""
            update_logs.append(f"{name:<20}: {old} -> {new}")

        if ms1_error is not None:
            _add_update_log("ms1_error", self.ms1_error, ms1_error)
            self.ms1_error = ms1_error
        if ms2_error is not None:
            _add_update_log("ms2_error", self.ms2_error, ms2_error)
            self.ms2_error = ms2_error
        if rt_error is not None:
            _add_update_log("rt_error", self.rt_error, rt_error)
            self.rt_error = rt_error
        if mobility_error is not None:
            _add_update_log("mobility_error", self.mobility_error, mobility_error)
            self.mobility_error = mobility_error
        if num_candidates is not None:
            _add_update_log("num_candidates", self.num_candidates, num_candidates)
            self.num_candidates = num_candidates
        if classifier_version is not None:
            _add_update_log(
                "classifier_version", self.classifier_version, classifier_version
            )
            self.classifier_version = classifier_version
        if fwhm_rt is not None:
            _add_update_log("fwhm_rt", self.fwhm_rt, fwhm_rt)
            self.fwhm_rt = fwhm_rt
        if fwhm_mobility is not None:
            _add_update_log("fwhm_mobility", self.fwhm_mobility, fwhm_mobility)
            self.fwhm_mobility = fwhm_mobility
        if score_cutoff is not None:
            _add_update_log("score_cutoff", self.score_cutoff, score_cutoff)
            self.score_cutoff = score_cutoff

        for log in update_logs:
            self.reporter.log_string(
                f"Updating optimization_manager: {log}", verbosity="info"
            )
