"""Configuration Module for Candidate Scoring."""

import logging

import numba as nb

from alphadia.constants.settings import MAX_FRAGMENT_MZ_TOLERANCE
from alphadia.search.jitclasses.jit_config import JITConfig

logger = logging.getLogger()


@nb.experimental.jitclass()
class CandidateScoringConfigJIT:
    collect_fragments: nb.boolean
    score_grouped: nb.boolean
    exclude_shared_ions: nb.boolean
    top_k_fragments: nb.uint32
    top_k_isotopes: nb.uint32
    reference_channel: nb.int16
    quant_window: nb.uint32
    quant_all: nb.boolean

    precursor_mz_tolerance: nb.float32
    fragment_mz_tolerance: nb.float32

    experimental_xic: nb.boolean

    def __init__(
        self,
        collect_fragments: nb.boolean,
        score_grouped: nb.boolean,
        exclude_shared_ions: nb.types.bool_,
        top_k_fragments: nb.uint32,
        top_k_isotopes: nb.uint32,
        reference_channel: nb.int16,
        quant_window: nb.uint32,
        quant_all: nb.boolean,
        precursor_mz_tolerance: nb.float32,
        fragment_mz_tolerance: nb.float32,
        experimental_xic: nb.boolean,
    ) -> None:
        """Numba JIT compatible config object for CandidateScoring.
        Will be emitted when `CandidateScoringConfig.jitclass()` is called.

        Please refer to :class:`.alphadia.scoring.CandidateScoringConfig` for documentation.
        """

        self.collect_fragments = collect_fragments
        self.score_grouped = score_grouped
        self.exclude_shared_ions = exclude_shared_ions
        self.top_k_fragments = top_k_fragments
        self.top_k_isotopes = top_k_isotopes
        self.reference_channel = reference_channel
        self.quant_window = quant_window
        self.quant_all = quant_all

        self.precursor_mz_tolerance = precursor_mz_tolerance
        self.fragment_mz_tolerance = fragment_mz_tolerance
        self.experimental_xic = experimental_xic


class CandidateScoringConfig(
    JITConfig
):  # TODO rename to CandidateScoringHyperparameters
    """Config object for CandidateScoring."""

    _jit_container_type = CandidateScoringConfigJIT

    def __init__(self):
        """Create default config for CandidateScoring"""
        super().__init__()

        self.collect_fragments = True
        self.score_grouped = False
        self.exclude_shared_ions = True
        self.top_k_fragments = 12
        self.top_k_isotopes = 4
        self.reference_channel = -1
        self.quant_window = 3
        self.quant_all = False
        self.precursor_mz_tolerance = 15
        self.fragment_mz_tolerance = 15
        self.experimental_xic = False

    @property
    def collect_fragments(self) -> bool:
        """Collect fragment features.
        Default: `collect_fragments = False`"""
        return self._collect_fragments

    @collect_fragments.setter
    def collect_fragments(self, value):
        self._collect_fragments = value

    @property
    def score_grouped(self) -> bool:
        """When multiplexing is used, some grouped features are calculated taking into account all channels.
        Default: `score_grouped = False`"""
        return self._score_grouped

    @score_grouped.setter
    def score_grouped(self, value):
        self._score_grouped = value

    @property
    def exclude_shared_ions(self) -> int:
        """When multiplexing is used, some fragments are shared for the same peptide with different labels.
        This setting removes fragments who are shared by more than one channel.
        Default: `exclude_shared_ions = True`"""
        return self._exclude_shared_ions

    @exclude_shared_ions.setter
    def exclude_shared_ions(self, value):
        self._exclude_shared_ions = value

    @property
    def top_k_fragments(self) -> int:
        """The number of fragments to consider for scoring. The top_k_fragments most intense fragments are used.
        Default: `top_k_fragments = 12`"""
        return self._top_k_fragments

    @top_k_fragments.setter
    def top_k_fragments(self, value):
        self._top_k_fragments = value

    @property
    def top_k_isotopes(self) -> int:
        """The number of precursor isotopes to consider for scoring. The first top_k_isotopes most intense isotopes are used.
        Default: `top_k_isotopes = 4`"""
        return self._top_k_isotopes

    @top_k_isotopes.setter
    def top_k_isotopes(self, value):
        self._top_k_isotopes = value

    @property
    def reference_channel(self) -> int:
        """When multiplexing is being used, a reference channel can be defined for calculating reference channel deopendent features.
        The channel information is used as defined in the `channel` column in the precursor dataframe. If set to -1, no reference channel is used.
        Default: `reference_channel = -1`"""
        return self._reference_channel

    @reference_channel.setter
    def reference_channel(self, value):
        self._reference_channel = value

    @property
    def quant_window(self) -> int:
        """The quantification window size in cycles.
        the area will be calculated from `scan_center - quant_window` to `scan_center + quant_window`.
        Default: `quant_window = 3`"""
        return self._quant_window

    @quant_window.setter
    def quant_window(self, value):
        self._quant_window = value

    @property
    def quant_all(self) -> bool:
        """Quantify all fragments in the quantification window.
        Default: `quant_all = False`"""
        return self._quant_all

    @quant_all.setter
    def quant_all(self, value):
        self._quant_all = value

    @property
    def precursor_mz_tolerance(self) -> float:
        """The precursor m/z tolerance in ppm.
        Default: `precursor_mz_tolerance = 10`"""
        return self._precursor_mz_tolerance

    @precursor_mz_tolerance.setter
    def precursor_mz_tolerance(self, value):
        self._precursor_mz_tolerance = value

    @property
    def fragment_mz_tolerance(self) -> float:
        """The fragment m/z tolerance in ppm.
        Default: `fragment_mz_tolerance = 15`"""
        return self._fragment_mz_tolerance

    @fragment_mz_tolerance.setter
    def fragment_mz_tolerance(self, value):
        self._fragment_mz_tolerance = value

    @property
    def experimental_xic(self) -> bool:
        """Use experimental XIC features.
        Default: `experimental_xic = False`"""
        return self._experimental_xic

    @experimental_xic.setter
    def experimental_xic(self, value):
        self._experimental_xic = value

    def validate(self):
        """Validate all properties of the config object.
        Should be called whenever a property is changed."""

        assert isinstance(self.score_grouped, bool), "score_grouped must be a boolean"
        assert self.top_k_fragments > 0, "top_k_fragments must be greater than 0"
        assert self.top_k_isotopes > 0, "top_k_isotopes must be greater than 0"
        assert (
            self.reference_channel >= -1
        ), "reference_channel must be greater than or equal to -1"
        # assert not (self.score_grouped == True and self.reference_channel == -1), 'for grouped scoring, reference_channel must be set to a valid channel'

        assert (
            self.precursor_mz_tolerance >= 0
        ), "precursor_mz_tolerance must be greater than or equal to 0"
        assert (
            self.precursor_mz_tolerance < 200
        ), "precursor_mz_tolerance must be less than 200"
        assert (
            self.fragment_mz_tolerance >= 0
        ), "fragment_mz_tolerance must be greater than or equal to 0"
        assert (
            self.fragment_mz_tolerance <= MAX_FRAGMENT_MZ_TOLERANCE
        ), f"fragment_mz_tolerance must be less than or equal {MAX_FRAGMENT_MZ_TOLERANCE}"


candidate_config_type = CandidateScoringConfigJIT.class_type.instance_type
