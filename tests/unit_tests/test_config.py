import os
from io import StringIO

import pytest
import yaml

from alphadia.workflow.config import Config

default_config = """
version: 1

general:
  thread_count: 10
  # maximum number of threads or processes to use per raw file
  reuse_calibration: false
  reuse_quant: false
  astral_ms1: false
  log_level: 'INFO'
  wsl: false
  mmap_detector_events: false
  use_gpu: true

library_loading:
  rt_heuristic: 180
  # if retention times are reported in absolute units, the rt_heuristic defines rt is interpreted as minutes or seconds

library_prediction:
  predict: False
  enzyme: trypsin
  fixed_modifications: 'Carbamidomethyl@C'
  variable_modifications: 'Oxidation@M'
  missed_cleavages: 1
  precursor_len:
    - 7
    - 35
  precursor_charge:
    - 2
    - 4
  """


config_1_yaml = """
general:
  thread_count: 10
  reuse_calibration: true
  reuse_quant: true
  use_gpu: true
  astral_ms1: false
  log_level: INFO
library_prediction:
  predict: false
  enzyme: trypsin
  fixed_modifications: Carbamidomethyl@C
  missed_cleavages: 2
  precursor_len:
    - 7
    - 6
  precursor_charge:
    - 2
    - 4
"""


config_2_yaml = """
version: 3
general:
  thread_count: 10
  reuse_calibration: true
  reuse_quant: true
  use_gpu: true
  astral_ms1: false
  log_level: INFO
library_prediction:
  predict: false
  enzyme: trypsin
  fixed_modifications: Carbamidomethyl@C
  missed_cleavages: 2
  precursor_len:
    - 2
    - 12
  precursor_charge:
    - 2
    - 4
"""

target_yaml = """
version: 3
general:
  thread_count: 10
  reuse_calibration: true
  reuse_quant: true
  astral_ms1: false
  log_level: INFO
  wsl: false
  mmap_detector_events: false
  use_gpu: true
library_loading:
  rt_heuristic: 180
library_prediction:
  predict: false
  enzyme: trypsin
  fixed_modifications: Carbamidomethyl@C
  variable_modifications: Oxidation@M
  missed_cleavages: 2
  precursor_len:
  - 2
  - 12
  precursor_charge:
  - 2
  - 4
"""

target_tsv = """	default	Experiment 1	Experiment 2
version	1	-	3
general.thread_count	10	-	-
general.reuse_calibration	False	True	-
general.reuse_quant	False	True	-
general.astral_ms1	False	-	-
general.log_level	INFO	-	-
general.wsl	False	-	-
general.mmap_detector_events	False	-	-
general.use_gpu	True	-	-
library_loading.rt_heuristic	180	-	-
library_prediction.predict	False	-	-
library_prediction.enzyme	trypsin	-	-
library_prediction.fixed_modifications	Carbamidomethyl@C	-	-
library_prediction.variable_modifications	Oxidation@M	-	-
library_prediction.missed_cleavages	1	2	-
library_prediction.precursor_len[0]	7	-	2
library_prediction.precursor_len[1]	35	6	12
library_prediction.precursor_charge[0]	2	-	-
library_prediction.precursor_charge[1]	4	-	-"""


# def test_update_function():
#     config_1 = Config("Experiment 1")
#     config_1.config = yaml.safe_load(StringIO(config_1_yaml))
#     config_2 = Config("Experiment 2")
#     config_2.config = yaml.safe_load(StringIO(config_2_yaml))
#     default = Config("default")
#     default.config = yaml.safe_load(StringIO(default_config))
#
#     default.update([config_1, config_2])
#
#     assert default.config == yaml.safe_load(StringIO(target_yaml))
#
#
# def test_get_modifications_table():
#     config_1 = Config("Experiment 1")
#     config_1.config = yaml.safe_load(StringIO(config_1_yaml))
#     config_2 = Config("Experiment 2")
#     config_2.config = yaml.safe_load(StringIO(config_2_yaml))
#     default = Config("default")
#     default.config = yaml.safe_load(StringIO(default_config))
#
#     table = get_update_table(default, [config_1, config_2])
#     tsv = table.to_csv(sep="\t")
#
#     table = pd.read_csv(StringIO(tsv), sep="\t", index_col=0)
#     target = pd.read_csv(StringIO(target_tsv), sep="\t", index_col=0)
#
#     pd.testing.assert_frame_equal(table, target)


generic_default_config = """
    simple_value_int: 1
    simple_value_float: 2.0
    simple_value_str: three
    nested_values:
        nested_value_1: 1
        nested_value_2: 2
    simple_list:
        - 1
        - 2
        - 3
    nested_list:
        - name: nested_list_value_1
          key1: value11
          key2: value21
          key3:
            - 311
            - 312
            - 313
        - name: nested_list_value_2
          key1: value12
          key2: value22
          key3:
            - 312
            - 322
            - 323
    """

expected_generic_default_config_dict = {
    "simple_value_int": 1,
    "simple_value_float": 2.0,
    "simple_value_str": "three",
    "nested_values": {"nested_value_1": 1, "nested_value_2": 2},
    "simple_list": [1, 2, 3],
    "nested_list": [
        {
            "name": "nested_list_value_1",
            "key1": "value11",
            "key2": "value21",
            "key3": [311, 312, 313],
        },
        {
            "name": "nested_list_value_2",
            "key1": "value12",
            "key2": "value22",
            "key3": [312, 322, 323],
        },
    ],
}


def test_config_update_empty_list():
    """Test updating a config with an empty list."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    # when
    config_1.update([])

    assert config_1.to_dict() == expected_generic_default_config_dict


def test_config_update_simple_two_files():
    """Test updating a config with simple values from two files."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict({"simple_value_int": 2, "simple_value_float": 4.0})

    config_3 = Config("second")
    config_3.from_dict(
        {
            "simple_value_float": 5.0,  # overwrites first
            "simple_value_str": "six",  # overwrites default
        }
    )

    # when
    config_1.update([config_2, config_3], print_modifications=True)

    assert config_1.to_dict() == expected_generic_default_config_dict | {
        "simple_value_int": 2,
        "simple_value_float": 5.0,
        "simple_value_str": "six",
    }


def test_config_update_advanced():
    """Test updating a config with nested values and lists"""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict(
        {
            "nested_values": {"nested_value_2": 42},
            "simple_list": [43, 44, 45, 999],
            "nested_list": [
                {
                    "name": "nested_list_value_2",
                    "key1": "46",
                    # "key2": ""
                    "key3": [
                        47,
                        48,
                    ],
                },
            ],
        }
    )

    # when
    config_1.update([config_2], print_modifications=True)

    assert config_1.to_dict() == expected_generic_default_config_dict | {
        "nested_values": {
            "nested_value_1": 1,  # original value
            "nested_value_2": 42,
        },
        "simple_list": [43, 44, 45, 999],  # item 999 added
        "nested_list": [
            {
                "name": "nested_list_value_2",
                "key1": "46",
                "key3": [
                    47,
                    48,
                ],
            },
        ],
    }


def test_config_update_new_key_raises():
    """Test updating a config with a new key."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict({"new_key": 0})

    # when
    with pytest.raises(ValueError, match="Key not found in target_dict: 'new_key'"):
        config_1.update([config_2], print_modifications=True)


def test_config_update_type_mismatch_raises():
    """Test updating a config with a different type"""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict({"simple_value_int": "one"})

    # when
    with pytest.raises(
        ValueError,
        match="Type mismatch for key 'simple_value_int': <class 'str'> != <class 'int'>",
    ):
        config_1.update([config_2], print_modifications=True)


def test_config_update_new_key_in_nested_list():
    """Test updating a config with a new item in a nested list."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict(
        {"nested_list": [{"key4": "value44", "name": "nested_list_value_3"}]}
    )

    # when
    config_1.update([config_2], print_modifications=True)

    assert config_1.to_dict() == expected_generic_default_config_dict | {
        "nested_list": [{"key4": "value44", "name": "nested_list_value_3"}],
    }


def test_config_update_nested_list_with_empty_list():
    """Test updating a config by giving an empty list."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict({"nested_list": []})

    # when
    config_1.update([config_2], print_modifications=True)

    assert config_1.to_dict() == expected_generic_default_config_dict | {
        "nested_list": []
    }


def test_config_update_default_config():
    """Test updating the default config with itself as a sanity check on that config."""

    config_base_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "alphadia", "constants", "default.yaml"
    )
    config_1 = Config("default")
    config_1.from_yaml(config_base_path)

    config_2 = Config("also_default")
    config_2.from_yaml(config_base_path)

    config_1.update([config_2], print_modifications=True)

    assert config_1.to_dict() == config_2.to_dict()
