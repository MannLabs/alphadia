"""Library prediction using peptdeep_kontext context-aware models."""

import logging
import os

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from peptdeep.pretrained_models import ModelManager as PeptDeepModelManager
from peptdeep_kontext.core.model_manager import (
    ModelManager as PeptDeepKontextModelManager,
)
from peptdeep_kontext.core.model_manager import ModelManagerConfig
from peptdeep_kontext.datasets.context import Context, ZeroContext
from peptdeep_kontext.datasets.prediction_aggregator import PredictionAggregator
from peptdeep_kontext.datasets.prediction_dataset import PredictionDataset
from peptdeep_kontext.utils.cfg import get_pretrained_model_dict_from_dir

from alphadia import utils
from alphadia.libtransform.base import ProcessingStep
from alphadia.libtransform.prediction import predict_charge_states
from alphadia.transferlearning import kontext

logger = logging.getLogger(__name__)


class PeptDeepKontextPrediction(ProcessingStep):
    def __init__(
        self,
        model_path: str,
        context_path: str | None = None,
        use_gpu: bool = True,
        fragment_types: list[str] | None = None,
        max_fragment_charge: int = 2,
        predict_charge: bool = False,
        min_charge_probability: float = 0.1,
        prediction_batch_size: int = 1000,
        config_path: str | None = None,
    ) -> None:
        """Predict RT and MS2 using peptdeep_kontext.

        Mobility and charge states are predicted with PeptDeep; RT and MS2 with the
        peptdeep_kontext downstream model, using a single context for all precursors.

        Parameters
        ----------
        model_path : str
            Path to the peptdeep_kontext ContextDownstream model directory.
        context_path : str, optional
            Path to a context.json learned by TTO. If not provided, the zero context is used.
        use_gpu : bool, optional
            Use GPU for prediction. Default is True.
        fragment_types : list[str], optional
            Fragment types to predict. Default is ["b", "y"].
        max_fragment_charge : int, optional
            Maximum fragment charge state to predict. Default is 2.
        predict_charge : bool, optional
            Whether to predict charge states using PeptDeep's charge model. Default is False.
        min_charge_probability : float, optional
            Minimum probability for including a predicted charge state. Default is 0.1.
        prediction_batch_size : int, optional
            Number of precursors per batch. Default is 1000.
        config_path : str, optional
            peptdeep_kontext pipeline config file whose `model_manager_config` section
            sets further prediction settings.
        """
        super().__init__()

        if fragment_types is None:
            fragment_types = ["b", "y"]

        self._use_gpu = use_gpu
        self._context_path = context_path
        self._predict_charge = predict_charge
        self._min_charge_probability = min_charge_probability
        self._charged_frag_types = get_charged_frag_types(
            fragment_types, max_fragment_charge
        )

        logger.info(
            f"Using peptdeep_kontext model {model_path} with context {context_path or 'zero context'}"
        )
        alphadia_settings = {"prediction_batch_size": prediction_batch_size}
        file_settings = kontext.load_config_file(config_path).get(
            "model_manager_config", {}
        )
        fixed_settings = {
            "pretrained_downstream_model": get_pretrained_model_dict_from_dir(
                model_path
            ),
            "requested_charged_fragment_types": self._charged_frag_types,
            "context_source_path": context_path,
            "dataset_config": {
                "feat_extractor": kontext.FEATURE_EXTRACTOR,
                "indicator_columns": None,
            },
        }
        self._config = kontext.build_config(
            ModelManagerConfig,
            alphadia_settings=alphadia_settings,
            file_settings=file_settings,
            fixed_settings=fixed_settings,
        )

    def validate(self, input: SpecLibBase) -> bool:
        return True

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        device = utils.get_torch_device(self._use_gpu)

        peptdeep_mgr = PeptDeepModelManager(device=device)

        precursor_df = input.precursor_df
        if self._predict_charge:
            logger.info("Predicting charge states with PeptDeep")
            precursor_df = predict_charge_states(
                peptdeep_mgr, precursor_df, self._min_charge_probability
            )

        logger.info("Predicting mobility with PeptDeep")
        precursor_df = peptdeep_mgr.predict_mobility(precursor_df)

        if self._context_path:
            if not os.path.isfile(self._context_path):
                raise FileNotFoundError(
                    f"peptdeep_kontext context {self._context_path} does not exist"
                )
            context = Context()
            context.load(self._context_path)
        else:
            context = ZeroContext()

        prediction_dataset = PredictionDataset(
            self._config.dataset_config, precursor_df, context
        )
        prediction_aggregator = PredictionAggregator(prediction_dataset.precursor_df)
        prediction_aggregator.reset()

        kontext_mgr = PeptDeepKontextModelManager(self._config)
        # ModelManager always picks CUDA if available, so respect general.use_gpu here
        kontext_mgr.device = kontext.get_torch_device(self._use_gpu)

        logger.info("Predicting RT and MS2 with peptdeep_kontext")
        kontext_mgr.predict(prediction_dataset, prediction_aggregator)

        return prediction_aggregator.predicted_spectral_library
