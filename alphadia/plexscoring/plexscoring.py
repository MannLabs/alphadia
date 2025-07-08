"""Main Implementation of Candidate Scoring System."""

import logging

import alphatims.utils
import numpy as np
import pandas as pd

from alphadia.data import alpharaw_wrapper, bruker
from alphadia.plexscoring import quadrupole
from alphadia.plexscoring.config import CandidateConfig
from alphadia.plexscoring.containers import ScoreGroupContainer
from alphadia.plexscoring.output import OutputPsmDF
from alphadia.plexscoring.utils import calculate_score_groups, merge_missing_columns
from alphadia.utilities.fragment_container import FragmentContainer
from alphadia.utils import (
    USE_NUMBA_CACHING,
    get_isotope_columns,
)
from alphadia.validation.schemas import (
    candidates_schema,
    features_schema,
    fragment_features_schema,
    fragments_flat_schema,
    precursors_flat_schema,
)

logger = logging.getLogger()


def _get_isotope_column_names(colnames):
    return [f"i_{i}" for i in get_isotope_columns(colnames)]


@alphatims.utils.pjit(cache=USE_NUMBA_CACHING)
def _executor(
    i,
    sg_container,
    psm_proto_df,
    fragment_container,
    jit_data,
    config,
    quadrupole_calibration,
    debug,
):
    """
    Helper function.
    Is decorated with alphatims.utils.pjit to enable parallel execution of HybridElutionGroup.process.
    """

    sg_container[i].process(
        psm_proto_df,
        fragment_container,
        jit_data,
        config,
        quadrupole_calibration,
        debug,
    )


class CandidateScoring:
    """Calculate features for each precursor candidate used in scoring."""

    def __init__(
        self,
        dia_data: bruker.TimsTOFTransposeJIT | alpharaw_wrapper.AlphaRawJIT,
        precursors_flat: pd.DataFrame,
        fragments_flat: pd.DataFrame,
        quadrupole_calibration: quadrupole.SimpleQuadrupole = None,
        config: CandidateConfig = None,
        rt_column: str = "rt_library",
        mobility_column: str = "mobility_library",
        precursor_mz_column: str = "mz_library",
        fragment_mz_column: str = "mz_library",
    ):
        """Initialize candidate scoring step.
        The features can then be used for scoring, calibration and quantification.

        Parameters
        ----------

        dia_data : data.TimsTOFTransposeJIT
            The raw mass spec data as a TimsTOFTransposeJIT object.

        precursors_flat : pd.DataFrame
            A DataFrame containing precursor information.
            The DataFrame will be validated by using the `alphadia.validation.schemas.precursors_flat` schema.

        fragments_flat : pd.DataFrame
            A DataFrame containing fragment information.
            The DataFrame will be validated by using the `alphadia.validation.schemas.fragments_flat` schema.

        quadrupole_calibration : quadrupole.SimpleQuadrupole, default=None
            An object containing the quadrupole calibration information.
            If None, an uncalibrated quadrupole will be used.
            The object musst have a `jit` method which returns a Numba JIT compiled instance of the calibration function.

        config : CandidateConfig, default = None
            A Numba jit compatible object containing the configuration for the candidate scoring.
            If None, the default configuration will be used.

        rt_column : str, default='rt_library'
            The name of the column in `precursors_flat` containing the RT information.
            This property needs to be changed to `rt_calibrated` if the data has been calibrated.

        mobility_column : str, default='mobility_library'
            The name of the column in `precursors_flat` containing the mobility information.
            This property needs to be changed to `mobility_calibrated` if the data has been calibrated.

        precursor_mz_column : str, default='mz_library'
            The name of the column in `precursors_flat` containing the precursor m/z information.
            This property needs to be changed to `mz_calibrated` if the data has been calibrated.

        fragment_mz_column : str, default='mz_library'
            The name of the column in `fragments_flat` containing the fragment m/z information.
            This property needs to be changed to `mz_calibrated` if the data has been calibrated.
        """

        self._dia_data = dia_data

        precursors_flat_schema.validate(precursors_flat, warn_on_critical_values=True)
        self.precursors_flat_df = precursors_flat

        fragments_flat_schema.validate(fragments_flat, warn_on_critical_values=True)
        self.fragments_flat = fragments_flat

        # check if a valid quadrupole calibration is provided
        if quadrupole_calibration is None:
            self.quadrupole_calibration = quadrupole.SimpleQuadrupole(dia_data.cycle)
        else:
            self.quadrupole_calibration = quadrupole_calibration

        # check if a valid config is provided
        if config is None:
            self.config = CandidateConfig()
        else:
            self.config = config

        self.rt_column = rt_column
        self.mobility_column = mobility_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column

    @property
    def dia_data(self):
        """Get the raw mass spec data as a TimsTOFTransposeJIT object."""
        return self._dia_data

    @property
    def precursors_flat_df(self):
        """Get the DataFrame containing precursor information."""
        return self._precursors_flat_df

    @precursors_flat_df.setter
    def precursors_flat_df(self, precursors_flat_df):
        precursors_flat_schema.validate(
            precursors_flat_df, warn_on_critical_values=True
        )
        self._precursors_flat_df = precursors_flat_df.sort_values(by="precursor_idx")

    @property
    def fragments_flat_df(self):
        """Get the DataFrame containing fragment information."""
        return self._fragments_flat

    @fragments_flat_df.setter
    def fragments_flat_df(self, fragments_flat):
        fragments_flat_schema.validate(fragments_flat, warn_on_critical_values=True)
        self._fragments_flat = fragments_flat

    @property
    def quadrupole_calibration(self):
        """Get the quadrupole calibration object."""
        return self._quadrupole_calibration

    @quadrupole_calibration.setter
    def quadrupole_calibration(self, quadrupole_calibration):
        if not hasattr(quadrupole_calibration, "jit"):
            raise AttributeError("quadrupole_calibration must have a jit method")
        self._quadrupole_calibration = quadrupole_calibration

    @property
    def config(self):
        """Get the configuration object."""
        return self._config

    @config.setter
    def config(self, config):
        config.validate()
        self._config = config

    def assemble_score_group_container(
        self, candidates_df: pd.DataFrame
    ) -> ScoreGroupContainer:
        """Assemble the Numba JIT compatible score group container from a candidate dataframe.

        If not present, the `rank` column will be added to the candidate dataframe.
        Then score groups are calculated using :func:`.calculate_score_groups` function.
        If configured in :attr:`.CandidateConfig.score_grouped`, all channels will be grouped into a single score group.
        Otherwise, each channel will be scored separately.

        The candidate dataframe is validated using the :func:`.validate.candidates` schema.

        Parameters
        ----------

        candidates_df : pd.DataFrame
            A DataFrame containing the candidates.

        Returns
        -------

        score_group_container : ScoreGroupContainer
            A Numba JIT compatible score group container.

        """

        precursor_columns = [
            "channel",
            "flat_frag_start_idx",
            "flat_frag_stop_idx",
            "charge",
            "decoy",
            "channel",
            self.precursor_mz_column,
        ] + _get_isotope_column_names(self.precursors_flat_df.columns)

        candidates_df = merge_missing_columns(
            candidates_df,
            self.precursors_flat_df,
            precursor_columns,
            on=["precursor_idx"],
            how="left",
        )

        # check if channel column is present
        if "channel" not in candidates_df.columns:
            candidates_df["channel"] = np.zeros(len(candidates_df), dtype=np.uint8)

        # check if monoisotopic abundance column is present
        if "i_0" not in candidates_df.columns:
            candidates_df["i_0"] = np.ones(len(candidates_df), dtype=np.float32)

        # calculate score groups
        candidates_df = calculate_score_groups(
            candidates_df, group_channels=self.config.score_grouped
        )

        # validate dataframe schema and prepare jitclass compatible dtypes
        candidates_schema.validate(candidates_df, warn_on_critical_values=True)

        score_group_container = ScoreGroupContainer()
        score_group_container.build_from_df(
            candidates_df["elution_group_idx"].values,
            candidates_df["score_group_idx"].values,
            candidates_df["precursor_idx"].values,
            candidates_df["channel"].values,
            candidates_df["rank"].values,
            candidates_df["flat_frag_start_idx"].values,
            candidates_df["flat_frag_stop_idx"].values,
            candidates_df["scan_start"].values,
            candidates_df["scan_stop"].values,
            candidates_df["scan_center"].values,
            candidates_df["frame_start"].values,
            candidates_df["frame_stop"].values,
            candidates_df["frame_center"].values,
            candidates_df["charge"].values,
            candidates_df[self.precursor_mz_column].values,
            candidates_df[_get_isotope_column_names(candidates_df.columns)].values,
        )

        return score_group_container

    def assemble_fragments(self) -> FragmentContainer:
        """Assemble the Numba JIT compatible fragment container from a fragment dataframe.

        If not present, the `cardinality` column will be added to the fragment dataframe and set to 1.
        Then the fragment dataframe is validated using the :func:`.validate.fragments_flat` schema.

        Returns
        -------

        fragment_container : fragments.FragmentContainer
            A Numba JIT compatible fragment container.
        """

        # set cardinality to 1 if not present
        if "cardinality" in self.fragments_flat.columns:
            pass

        else:
            logger.warning(
                "Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1."
            )
            self.fragments_flat["cardinality"] = np.ones(
                len(self.fragments_flat), dtype=np.uint8
            )

        # validate dataframe schema and prepare jitclass compatible dtypes
        fragments_flat_schema.validate(
            self.fragments_flat, warn_on_critical_values=True
        )

        return FragmentContainer(
            self.fragments_flat["mz_library"].values,
            self.fragments_flat[self.fragment_mz_column].values,
            self.fragments_flat["intensity"].values,
            self.fragments_flat["type"].values,
            self.fragments_flat["loss_type"].values,
            self.fragments_flat["charge"].values,
            self.fragments_flat["number"].values,
            self.fragments_flat["position"].values,
            self.fragments_flat["cardinality"].values,
        )

    def collect_candidates(
        self, candidates_df: pd.DataFrame, psm_proto_df
    ) -> pd.DataFrame:
        """Collect the features from the score group container and return a DataFrame.

        Parameters
        ----------

        score_group_container : ScoreGroupContainer
            A Numba JIT compatible score group container.

        candidates_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        Returns
        -------

        candidates_psm_df : pd.DataFrame
            A DataFrame containing the features for each candidate.
        """

        feature_columns = [
            "base_width_mobility",
            "base_width_rt",
            "rt_observed",
            "mobility_observed",
            "mono_ms1_intensity",
            "top_ms1_intensity",
            "sum_ms1_intensity",
            "weighted_ms1_intensity",
            "weighted_mass_deviation",
            "weighted_mass_error",
            "mz_observed",
            "mono_ms1_height",
            "top_ms1_height",
            "sum_ms1_height",
            "weighted_ms1_height",
            "isotope_intensity_correlation",
            "isotope_height_correlation",
            "n_observations",
            "intensity_correlation",
            "height_correlation",
            "intensity_fraction",
            "height_fraction",
            "intensity_fraction_weighted",
            "height_fraction_weighted",
            "mean_observation_score",
            "sum_b_ion_intensity",
            "sum_y_ion_intensity",
            "diff_b_y_ion_intensity",
            "f_masked",
            "fragment_scan_correlation",
            "template_scan_correlation",
            "fragment_frame_correlation",
            "top3_frame_correlation",
            "template_frame_correlation",
            "top3_b_ion_correlation",
            "n_b_ions",
            "top3_y_ion_correlation",
            "n_y_ions",
            "cycle_fwhm",
            "mobility_fwhm",
            "delta_frame_peak",
            "top_3_ms2_mass_error",
            "mean_ms2_mass_error",
            "n_overlapping",
            "mean_overlapping_intensity",
            "mean_overlapping_mass_error",
        ]

        precursor_idx, rank, features = psm_proto_df.to_precursor_df()

        df = pd.DataFrame(features, columns=feature_columns)
        df["precursor_idx"] = precursor_idx
        df["rank"] = rank

        # join candidate columns
        candidate_df_columns = [
            "elution_group_idx",
            "scan_center",
            "scan_start",
            "scan_stop",
            "frame_center",
            "frame_start",
            "frame_stop",
        ]

        candidate_df_columns += ["score"] if "score" in candidates_df.columns else []

        df = merge_missing_columns(
            df,
            candidates_df,
            candidate_df_columns,
            on=["precursor_idx", "rank"],
            how="left",
        )

        # join precursor columns
        precursor_df_columns = [
            "rt_library",
            "mobility_library",
            "mz_library",
            "charge",
            "decoy",
            "channel",
            "flat_frag_start_idx",
            "flat_frag_stop_idx",
            "proteins",
            "genes",
            "sequence",
            "mods",
            "mod_sites",
        ] + _get_isotope_column_names(self.precursors_flat_df.columns)

        precursor_df_columns += (
            [self.rt_column] if self.rt_column not in precursor_df_columns else []
        )
        precursor_df_columns += (
            [self.mobility_column]
            if self.mobility_column not in precursor_df_columns
            else []
        )
        precursor_df_columns += (
            [self.precursor_mz_column]
            if self.precursor_mz_column not in precursor_df_columns
            else []
        )

        df = merge_missing_columns(
            df,
            self.precursors_flat_df,
            precursor_df_columns,
            on=["precursor_idx"],
            how="left",
        )

        # calculate delta_rt

        if self.rt_column == "rt_library":
            df["delta_rt"] = df["rt_observed"] - df["rt_library"]
        else:
            df["delta_rt"] = df["rt_observed"] - df[self.rt_column]

        # calculate number of K in sequence
        df["n_K"] = df["sequence"].str.count("K")
        df["n_R"] = df["sequence"].str.count("R")
        df["n_P"] = df["sequence"].str.count("P")

        return df

    def collect_fragments(
        self, candidates_df: pd.DataFrame, psm_proto_df
    ) -> pd.DataFrame:
        """Collect the fragment-level features from the score group container and return a DataFrame.

        Parameters
        ----------

        score_group_container : ScoreGroupContainer
            A Numba JIT compatible score group container.

        candidates_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        Returns
        -------

        fragment_psm_df : pd.DataFrame
            A DataFrame containing the features for each fragment.

        """

        colnames = [
            "precursor_idx",
            "rank",
            "mz_library",
            "mz",
            "mz_observed",
            "height",
            "intensity",
            "mass_error",
            "correlation",
            "position",
            "number",
            "type",
            "charge",
            "loss_type",
        ]
        df = pd.DataFrame(
            {
                key: value
                for value, key in zip(
                    psm_proto_df.to_fragment_df(), colnames, strict=True
                )
            }
        )

        # join precursor columns
        precursor_df_columns = [
            "elution_group_idx",
            "decoy",
        ]
        df = merge_missing_columns(
            df,
            self.precursors_flat_df,
            precursor_df_columns,
            on=["precursor_idx"],
            how="left",
        )

        return df

    def __call__(
        self,
        candidates_df,
        thread_count=10,
        debug=False,
        include_decoy_fragment_features=False,
    ):
        """Calculate features for each precursor candidate used for scoring.

        Parameters
        ----------

        candidates_df : pd.DataFrame
            A DataFrame containing the candidates.

        thread_count : int, default=10
            The number of threads to use for parallel processing.

        debug : bool, default=False
            Process only the first 10 elution groups and display full debug information.

        include_decoy_fragment_features : bool, default=False
            Include fragment features for decoy candidates.

        Returns
        -------

        candidate_features_df : pd.DataFrame
            A DataFrame containing the features for each candidate.

        fragment_features_df : pd.DataFrame
            A DataFrame containing the features for each fragment.

        """
        logger.info("Starting candidate scoring")

        fragment_container = self.assemble_fragments()

        candidates_schema.validate(candidates_df, warn_on_critical_values=True)

        score_group_container = self.assemble_score_group_container(candidates_df)
        n_candidates = score_group_container.get_candidate_count()
        psm_proto_df = OutputPsmDF(n_candidates, self.config.top_k_fragments)

        iterator_len = len(score_group_container)

        if debug:
            logger.info("Debug mode enabled. Processing only the first 10 score groups")
            thread_count = 1
            iterator_len = min(10, iterator_len)

        alphatims.utils.set_threads(thread_count)
        _executor(
            range(iterator_len),
            score_group_container,
            psm_proto_df,
            fragment_container,
            self.dia_data,
            self.config.jitclass(),
            self.quadrupole_calibration.jit,
            debug,
        )

        logger.info("Finished candidate processing")
        logger.info("Collecting candidate features")
        candidate_features_df = self.collect_candidates(candidates_df, psm_proto_df)
        features_schema.validate(candidate_features_df, warn_on_critical_values=True)

        logger.info("Collecting fragment features")
        fragment_features_df = self.collect_fragments(candidates_df, psm_proto_df)
        fragment_features_schema.validate(
            fragment_features_df, warn_on_critical_values=True
        )

        logger.info("Finished candidate scoring")

        del score_group_container
        del fragment_container

        return candidate_features_df, fragment_features_df
