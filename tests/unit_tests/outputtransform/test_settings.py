from conftest import make_config

from alphadia.constants.keys import NormalizationMethods
from alphadia.outputtransform.settings import LfqSettings, OutputSettings


def test_output_settings_parses_default_config():
    # given
    config = make_config()

    # when
    settings = OutputSettings.from_config(config)

    # then
    assert settings.file_format == config["search_output"]["file_format"]
    assert settings.fdr.fdr == config["fdr"]["fdr"]
    assert settings.lfq.min_k_fragments == config["search_output"]["min_k_fragments"]
    assert settings.lfq.thread_count == config["general"]["thread_count"]
    assert settings.transfer_learning.epochs == config["transfer_learning"]["epochs"]
    assert settings.transfer_library.charged_frag_types == [
        "b_z1",
        "b_z2",
        "y_z1",
        "y_z2",
    ]
    assert settings.channels == [0]


def test_lfq_settings_lowercases_normalization_method():
    # given
    config = make_config({"search_output": {"normalization_method": "QuantSelect"}})

    # when
    settings = LfqSettings.from_config(config)

    # then
    assert settings.normalization_method == NormalizationMethods.QUANTSELECT
