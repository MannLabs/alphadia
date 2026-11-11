import logging

from alphadia_search_rs import OptimizationContext

from alphadia.workflow.config import Config
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()

# tolerances that the next raw file of a search step can start from, see `calibration.warm_start`
WARM_START_TOLERANCES = ("ms1_error", "ms2_error", "mobility_error")


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
        initial_tolerances: dict[str, float] | None = None,
        **kwargs,
    ):
        """Manager of the search parameters that the optimization changes.

        Parameters
        ----------
        config : Config, optional
            Configuration, `search_initial` gives the initial values.
        gradient_length : float, optional
            Gradient length in seconds, for an initial retention time tolerance given as a proportion of it.
        path : str, optional
            Path to save the manager to and load it from.
        load_from_file : bool, default=True
            Load the manager from `path` if the file exists.
        initial_tolerances : dict[str, float], optional
            Initial values of `WARM_START_TOLERANCES` (e.g. from previous raw files) that replace narrower
            values only: a tolerance never starts wider than `search_initial`.
        """
        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        if not self.is_loaded_from_file:
            self.ms1_error = config["search_initial"]["ms1_tolerance"]
            self.ms2_error = config["search_initial"]["ms2_tolerance"]

            initial_rt_tolerance = config["search_initial"]["rt_tolerance"]
            self.rt_error = (
                initial_rt_tolerance
                if initial_rt_tolerance > 1
                else initial_rt_tolerance * gradient_length
            )
            max_rt_tolerance = config["search_initial"].get("max_rt_tolerance")
            if max_rt_tolerance is not None:
                self.rt_error = min(self.rt_error, max_rt_tolerance)
            self.mobility_error = config["search_initial"]["mobility_tolerance"]

            for key, value in (initial_tolerances or {}).items():
                if key not in WARM_START_TOLERANCES:
                    raise ValueError(f"Unknown initial tolerance: {key}")
                setattr(self, key, min(getattr(self, key), value))

            self.num_candidates = config["search_initial"]["num_candidates"]

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

    def get_optimization_context(self) -> OptimizationContext:
        """Snapshot of the current optimization context, which later updates of this manager do not change."""
        return OptimizationContext(
            rt_error=self.rt_error,
            ms1_error=self.ms1_error,
            ms2_error=self.ms2_error,
            mobility_error=self.mobility_error,
            fwhm_rt=self.fwhm_rt,
            fwhm_mobility=self.fwhm_mobility,
            num_candidates=self.num_candidates,
            score_cutoff=self.score_cutoff,
        )
