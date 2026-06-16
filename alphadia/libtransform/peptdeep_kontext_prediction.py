"""Library prediction using peptdeep_kontext context-aware models."""

import logging

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from peptdeep.pretrained_models import ModelManager as PeptDeepModelManager
from peptdeep_kontext.core.model_manager import ModelManager as PeptDeepKontextModelManager
from peptdeep_kontext.core.model_manager import ModelManagerConfig
from peptdeep_kontext.datasets.context import Context, ZeroContext
from peptdeep_kontext.datasets.prediction_aggregator import PredictionAggregator
from peptdeep_kontext.datasets.prediction_dataset import (
    PredictionDataset,
    PredictionDatasetConfig,
)
from peptdeep_kontext.utils.cfg import get_pretrained_model_config_from_dir

from alphadia import utils
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger(__name__)


class PeptDeepKontextPrediction(ProcessingStep):
    def __init__(
        self,
        use_gpu: bool = True,
        peptdeep_kontext_model_path: str | None = None,
        context_path: str | None = None,
        fragment_types: list[str] | None = None,
        max_fragment_charge: int = 2,
        predict_charge: bool = False,
        min_charge_probability: float = 0.1,
        indicator_columns: list[str] | None = None,
    ) -> None:
        """Predict RT and MS2 using peptdeep_kontext context-aware models.

        Mobility is predicted with PeptDeep; RT and MS2 use the pretrained
        peptdeep_kontext downstream model conditioned on extracted context vectors.

        Parameters
        ----------
        use_gpu : bool, optional
            Use GPU for prediction. Default is True.
        peptdeep_kontext_model_path : str, optional
            Path to the pretrained peptdeep_kontext downstream model directory.
            If not provided, the default bundled model is used.
        context_path : str, optional
            Base path to the extracted context file (without ``.json`` extension).
            If not provided, a zero context is used.
        fragment_types : list[str], optional
            Fragment types to predict. Default is ["b", "y"].
        max_fragment_charge : int, optional
            Maximum fragment charge state to predict. Default is 2.
        predict_charge : bool, optional
            Whether to predict charge states using PeptDeep's charge model.
            Default is False.
        min_charge_probability : float, optional
            Minimum probability threshold for including a charge state.
            Default is 0.1.
        indicator_columns : list[str], optional
            Columns used to match precursors to context vectors (e.g.
            ``['raw_name']``). Defaults to ``['constant_context_indicator']``.
        """
        if fragment_types is None:
            fragment_types = ["b", "y"]
        if indicator_columns is None:
            indicator_columns = ["constant_context_indicator"]

        super().__init__()

        logger.info(f"Loading peptdeep_kontext model with context path {context_path}")

        self.use_gpu = use_gpu
        self.peptdeep_kontext_model_path = peptdeep_kontext_model_path
        self.context_path = context_path

        self.fragment_types = fragment_types
        self.max_fragment_charge = max_fragment_charge

        self.predict_charge = predict_charge
        self.min_charge_probability = min_charge_probability
        self.charged_frag_types = get_charged_frag_types(
            self.fragment_types, self.max_fragment_charge
        )
        self.model_mgr_config = ModelManagerConfig(
            pretrained_downstream_model=get_pretrained_model_config_from_dir(
                peptdeep_kontext_model_path
            ),
            requested_charged_fragment_types=self.charged_frag_types,
            dataset_config=PredictionDatasetConfig(
                feat_extractor="BertaFeatureExtractor",
                indicator_columns=indicator_columns,
            ),
        )

    def validate(self, input: list[str]) -> bool:
        return True

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        input.charged_frag_types = self.charged_frag_types

        device = utils.get_torch_device(self.use_gpu)

        # Use PeptDeep for mobility prediction only
        peptdeep_mgr = PeptDeepModelManager(device=device)

        precursor_df = input.precursor_df

        if self.predict_charge:
            charge_range = peptdeep_mgr.charge_model.charge_range
            min_supported = int(charge_range.min())
            max_supported = int(charge_range.max())

            if "charge" in precursor_df.columns:
                min_charge = max(min_supported, int(precursor_df["charge"].min()))
                max_charge = min(max_supported, int(precursor_df["charge"].max()))
            else:
                min_charge = min_supported
                max_charge = max_supported

            logger.info(
                f"Predicting charge states (charge range: {min_charge}-{max_charge}, "
                f"min probability: {self.min_charge_probability})"
            )
            n_before = len(precursor_df)
            precursor_df = peptdeep_mgr.predict_charge(
                precursor_df,
                min_precursor_charge=min_charge,
                max_precursor_charge=max_charge,
                charge_prob_cutoff=self.min_charge_probability,
            )
            n_dropped = n_before - len(precursor_df)
            logger.info(
                f"Charge prediction kept {len(precursor_df)} precursors, "
                f"{n_dropped} dropped by min_charge_probability filter"
            )

        logger.info("Predicting mobility with PeptDeep")
        precursor_df = peptdeep_mgr.predict_mobility(precursor_df)

        # Propagate charge/mobility updates before building the prediction dataset
        input._precursor_df = precursor_df

        if self.context_path:
            context = Context()
            context.load(f"{self.context_path}.json")
        else:
            context = ZeroContext()

        prediction_dataset = PredictionDataset(
            self.model_mgr_config.dataset_config,
            input.precursor_df,
            context,
        )

        prediction_aggregator = PredictionAggregator(input.precursor_df)
        prediction_aggregator.reset()

        kontext_mgr = PeptDeepKontextModelManager(
            model_manager_config=self.model_mgr_config
        )
        kontext_mgr.predict(prediction_dataset, prediction_aggregator)

        return prediction_aggregator.predicted_spectral_library
