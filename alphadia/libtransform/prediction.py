import logging
import os

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from peptdeep.pretrained_models import ModelManager as PeptDeepModelManager

from alphadia import utils
from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger(__name__)


def predict_charge_states(
    model_mgr: PeptDeepModelManager,
    precursor_df,
    min_charge_probability: float,
):
    """Predict charge states and filter by minimum probability threshold.

    Uses the model manager's charge model to expand the precursor dataframe
    with predicted charge states, dropping those below `min_charge_probability`.
    The charge range is clamped to the range already present in `precursor_df`
    when a ``charge`` column exists.
    """
    charge_range = model_mgr.charge_model.charge_range
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
        f"min probability: {min_charge_probability})"
    )
    n_before = len(precursor_df)
    precursor_df = model_mgr.predict_charge(
        precursor_df,
        min_precursor_charge=min_charge,
        max_precursor_charge=max_charge,
        charge_prob_cutoff=min_charge_probability,
    )
    n_dropped = n_before - len(precursor_df)
    logger.info(
        f"Charge prediction kept {len(precursor_df)} precursors, "
        f"{n_dropped} dropped by min_charge_probability filter"
    )
    return precursor_df


class PeptDeepPrediction(ProcessingStep):
    def __init__(
        self,
        use_gpu: bool = True,
        mp_process_num: int = 8,
        fragment_mz: list[int] | None = None,
        nce: int = 25,
        instrument: str = "Lumos",
        peptdeep_model_path: str | None = None,
        peptdeep_model_type: str | None = None,
        fragment_types: list[str] | None = None,
        max_fragment_charge: int = 2,
        predict_charge: bool = False,
        min_charge_probability: float = 0.1,
    ) -> None:
        """Predict RT, MS2 and mobility using PeptDeep.

        Parameters
        ----------
        use_gpu : bool, optional
            Use GPU for prediction. Default is True.
        mp_process_num : int, optional
            Number of processes to use for prediction. Default is 8.
        fragment_mz : list[int], optional
            MZ range for fragment prediction. Default is [100, 2000].
        nce : int, optional
            Normalized collision energy for prediction. Default is 25.
        instrument : str, optional
            Instrument type for prediction. Default is "Lumos". Must be a valid
            PeptDeep instrument.
        peptdeep_model_path : str, optional
            Path to a folder containing PeptDeep models. If not provided, the
            default models are used.
        peptdeep_model_type : str, optional
            Named model set to load via the PeptDeep model manager (e.g.
            ``"generic"``, ``"phospho"``, ``"digly"``). Default is None.
        fragment_types : list[str], optional
            Fragment types to predict. Default is ["b", "y"].
        max_fragment_charge : int, optional
            Maximum charge state to predict. Default is 2.
        predict_charge : bool, optional
            Whether to predict charge states using PeptDeep's charge model.
            Default is False.
        min_charge_probability : float, optional
            Minimum probability threshold for including a charge state.
            Default is 0.1.
        """
        if fragment_types is None:
            fragment_types = ["b", "y"]
        if fragment_mz is None:
            fragment_mz = [100, 2000]
        super().__init__()
        self.use_gpu = use_gpu
        self.fragment_mz = fragment_mz
        self.nce = nce
        self.instrument = instrument
        self.mp_process_num = mp_process_num
        self.peptdeep_model_path = peptdeep_model_path
        self.peptdeep_model_type = peptdeep_model_type

        self.fragment_types = fragment_types
        self.max_fragment_charge = max_fragment_charge

        self.predict_charge = predict_charge
        self.min_charge_probability = min_charge_probability

    def validate(self, input: list[str]) -> bool:
        return True

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        charged_frag_types = get_charged_frag_types(
            self.fragment_types, self.max_fragment_charge
        )

        input.charged_frag_types = charged_frag_types

        device = utils.get_torch_device(self.use_gpu)

        model_mgr = PeptDeepModelManager(device=device)
        # Set the requested charged fragment types for the ms2 model
        model_mgr.reinitialize_ms2_model(charged_frag_types=charged_frag_types)

        if self.peptdeep_model_type:
            logger.info(f"Loading PeptDeep models of type {self.peptdeep_model_type}")
            model_mgr.load_installed_models(self.peptdeep_model_type)
        else:
            logger.info("Using PeptDeep default model.")

        if self.peptdeep_model_path:
            if not os.path.exists(self.peptdeep_model_path):
                raise ValueError(
                    f"PeptDeep model checkpoint folder {self.peptdeep_model_path} does not exist"
                )

            logger.info(f"Loading PeptDeep models from {self.peptdeep_model_path}")

            model_mgr.load_external_models(
                ms2_model_file=os.path.join(self.peptdeep_model_path, "ms2.pth"),
                rt_model_file=os.path.join(self.peptdeep_model_path, "rt.pth"),
                ccs_model_file=os.path.join(self.peptdeep_model_path, "ccs.pth"),
                charge_model_file=os.path.join(self.peptdeep_model_path, "charge.pth"),
            )

        model_mgr.nce = self.nce
        model_mgr.instrument = self.instrument

        precursor_df = input.precursor_df

        if self.predict_charge:
            precursor_df = predict_charge_states(
                model_mgr, precursor_df, self.min_charge_probability
            )

        logger.info("Predicting RT, MS2 and mobility")
        res = model_mgr.predict_all(
            precursor_df,
            predict_items=["rt", "ms2", "mobility"],
            frag_types=charged_frag_types,
            process_num=self.mp_process_num,
        )

        if "fragment_mz_df" in res:
            logger.info("Adding fragment mz information")
            input._fragment_mz_df = res["fragment_mz_df"][charged_frag_types]

        if "fragment_intensity_df" in res:
            logger.info("Adding fragment intensity information")
            input._fragment_intensity_df = res["fragment_intensity_df"][
                charged_frag_types
            ]

        if "precursor_df" in res:
            logger.info("Adding precursor information")
            input._precursor_df = res["precursor_df"]

        return input
