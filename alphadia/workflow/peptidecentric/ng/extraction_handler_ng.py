import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

# TODO: these imports could be conditional: HybridCandidateConfig, HybridCandidateSelection, CandidateConfig, CandidateScoring
from alphadia.raw_data import DiaData
from alphadia.workflow.peptidecentric.extraction_handler import ClassicExtractionHandler
from alphadia.workflow.peptidecentric.ng.ng_mapper import (
    parse_candidates,
    speclib_to_ng,
)


class NgExtractionHandler(ClassicExtractionHandler):
    def _select_candidates(
        self, dia_data: DiaData, spectral_library: SpecLibFlat
    ) -> pd.DataFrame:
        """Select candidates using NG backend.

        See superclass documentation for interface details.
        """
        from alpha_ng import PeakGroupScoring, ScoringParameters

        # TODO this is a hack that needs to go once we don't need the "classic" dia_data object anymore
        dia_data_ng: DiaDataNG = dia_data[1]  # noqa: F821
        dia_data: DiaData = dia_data[0]

        # TODO needs to be stored
        speclib_ng = speclib_to_ng(
            spectral_library,
            rt_column=self._column_name_handler.get_rt_column(),
            mobility_column=self._column_name_handler.get_mobility_column(),
            precursor_mz_column=self._column_name_handler.get_precursor_mz_column(),
            fragment_mz_column=self._column_name_handler.get_fragment_mz_column(),
        )

        scoring_params = ScoringParameters()
        scoring_params.update(
            {
                "fwhm_rt": self._optimization_manager.fwhm_rt,  # 3.0,
                # 'kernel_size': 20,  # 15?
                "peak_length": self._config["search"]["quant_window"],
                "mass_tolerance": self._optimization_manager.ms2_error,
                "rt_tolerance": self._optimization_manager.rt_error,
                "candidate_count": self._optimization_manager.num_candidates,
            }
        )

        self._reporter.log_string(
            f"Using parameters: fwhm_rt={scoring_params.fwhm_rt}, "
            f"kernel_size={scoring_params.kernel_size}, "
            f"peak_length={scoring_params.peak_length}, "
            f"mass_tolerance={scoring_params.mass_tolerance}, "
            f"rt_tolerance={scoring_params.rt_tolerance}"
        )

        peak_group_scoring = PeakGroupScoring(scoring_params)

        candidates = peak_group_scoring.search_next_gen(dia_data_ng, speclib_ng)

        return parse_candidates(dia_data, candidates, spectral_library.precursor_df)

    def _score_candidates(
        self,
        candidates_df: pd.DataFrame,
        dia_data: DiaData,
        spectral_library: SpecLibFlat,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO this is a hack that needs to go once we don't need the "classic" dia_data object anymore
        return super()._score_candidates(candidates_df, dia_data[0], spectral_library)
