"""
A module that uses peptdeeepptcm to extract the context of raw files using peptdeepptcm. This is teh alternative to the transfer learning module.
"""
import logging

import numpy as np
import pandas as pd
import torch

from peptdeepptcm.core.tto import TTOConfig,TTOManager
from peptdeepptcm.models.model import PretrainedModelConfig
from peptdeepptcm.datasets.dataset import DatasetConfig
from peptdeepptcm.datasets.libloader import LibLoaderConfig
from peptdeepptcm.utils.cfg import get_pretrained_model_config_from_dir



class ContextExtractor:
    def __init__(self, 
                annotated_speclib_path: str, 
                charged_frag_types: list[str],
                pretrained_context_model_path: str,
                tto_epoch: int = 10,
                tto_batch_size: int = 1000,
                tto_lr: float = 0.1,
                tto_warmup_epochs: int = 5,
                context_indicator_columns: list[str] = ['raw_name'],
                verbose: bool = False) -> None:
        """
        Initialize the ContextExtractor.

        Parameters
        ----------
        annotated_speclib_path : str
            The path to the annotated spectral library.
        tto_epoch : int, optional
            The number of epochs for test time optimization, by default 10.
        tto_batch_size : int, optional
            The batch size for test time optimization, by default 1000.
        tto_lr : float, optional
            The learning rate for test time optimization, by default 0.1.
        tto_warmup_epochs : int, optional
            The number of warmup epochs for test time optimization, by default 5.
        verbose : bool, optional
            Whether to print verbose output during test time optimization, by default False.
        pretrained_context_model_path : str, optional
            The path to the pretrained context model, by default None.
        """
        tto_config = TTOConfig(
            tto_epochs=tto_epoch,
            tto_batch_size=tto_batch_size,
            tto_lr=tto_lr,
            tto_warmup_epochs=tto_warmup_epochs,
            verbose=verbose,
            pretrained_context_model=get_pretrained_model_config_from_dir(pretrained_context_model_path),
            tto_dataset_config=DatasetConfig(
                feat_extractor="BertaFeatureExtractor",
                indicator_columns=context_indicator_columns,
            ),
            tto_lib_loader_config=LibLoaderConfig(
                root_dirs=[annotated_speclib_path],
                frag_types=charged_frag_types,
            )
        )
        self.tto_manager = TTOManager(tto_config)

    def run(self, save_path: str) -> pd.DataFrame:
        """
        Extract the context of the raw files.

        Parameters
        ----------
        save_path : str
            The path to save the extracted context dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the extracted context.
        """
        self.tto_manager.run(save_path+".json")