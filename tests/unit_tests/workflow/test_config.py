import os
import tempfile
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from alphadia.exceptions import KeyAddedConfigError, TypeMismatchConfigError
from alphadia.workflow.config import Config

generic_default_config = """
    simple_value_int: 1
    simple_value_float: 2.0
    simple_value_str: three
    simple_value_bool: true
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
    "simple_value_bool": True,
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
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)))

    # when
    config_1.update([])

    assert config_1 == expected_generic_default_config_dict


def test_config_update_simple_two_files():
    """Test updating a config with simple values from two files."""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config({"simple_value_int": 2, "simple_value_float": 4.0}, "first")

    config_3 = Config(
        {
            "simple_value_float": 5.0,  # overwrites first
            "simple_value_str": "six",  # overwrites default
        },
        "second",
    )

    # when
    config_1.update([config_2, config_3], do_print=True)

    assert config_1 == expected_generic_default_config_dict | {
        "simple_value_int": 2,
        "simple_value_float": 5.0,
        "simple_value_str": "six",
    }


def test_config_update_advanced():
    """Test updating a config with nested values and lists"""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config(
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
        },
        "first",
    )

    # when
    config_1.update([config_2], do_print=True)

    assert config_1 == expected_generic_default_config_dict | {
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


def test_config_update_advanced_add_nested_list_item():
    """Test updating a config with nested values and lists, with new list longer than old one."""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config(
        {
            "nested_values": {"nested_value_2": 42},
            "simple_list": [43, 44, 45, 999],
            "nested_list": [
                {
                    "name": "nested_list_value_1",
                    "key1": "1",
                },
                {
                    "name": "nested_list_value_2",
                    "key1": "2",
                },
                {
                    "name": "nested_list_value_3",
                    "key1": "3",
                },
            ],
        },
        "first",
    )

    # when
    config_1.update([config_2], do_print=True)

    assert config_1 == expected_generic_default_config_dict | {
        "nested_values": {
            "nested_value_1": 1,  # original value
            "nested_value_2": 42,
        },
        "simple_list": [43, 44, 45, 999],  # item 999 added
        "nested_list": [
            {
                "name": "nested_list_value_1",
                "key1": "1",
            },
            {
                "name": "nested_list_value_2",
                "key1": "2",
            },
            {
                "name": "nested_list_value_3",
                "key1": "3",
            },
        ],
    }


def test_config_update_new_key_raises():
    """Test updating a config with a new key."""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config({"new_key": 0}, "first")

    # when
    with pytest.raises(KeyAddedConfigError):
        config_1.update([config_2], do_print=True)


@patch("alphadia.workflow.config.TOLERATED_KEYS", ["nested_values.new_key2"])
def test_config_update_new_key_tolerated():
    """Test updating a config with a new key that is tolerated."""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config({"nested_values": {"new_key2": 0}}, "first")

    # when
    config_1.update([config_2], do_print=True)

    assert config_1 == expected_generic_default_config_dict


def test_config_update_type_mismatch_raises():
    """Test updating a config with a different type"""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config({"simple_value_int": "one"}, "first")

    # when
    with pytest.raises(
        TypeMismatchConfigError,
    ):
        config_1.update([config_2], do_print=True)


def test_config_update_no_type_mismatch_on_boolean_string():
    """Test updating a config with a boolean that is a string works"""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config({"simple_value_bool": "false"}, "first")

    # when
    config_1.update([config_2], do_print=True)

    assert config_1 == expected_generic_default_config_dict | {
        "simple_value_bool": False,
    }


def test_config_update_new_key_in_nested_list():
    """Test updating a config with a new item in a nested list."""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config(
        {"nested_list": [{"key4": "value44", "name": "nested_list_value_3"}]}, "first"
    )

    # when
    config_1.update([config_2], do_print=True)

    assert config_1 == expected_generic_default_config_dict | {
        "nested_list": [{"key4": "value44", "name": "nested_list_value_3"}],
    }


def test_config_update_nested_list_with_empty_list():
    """Test updating a config by giving an empty list."""
    config_1 = Config(yaml.safe_load(StringIO(generic_default_config)), "default")

    config_2 = Config({"nested_list": []}, "first")

    # when
    config_1.update([config_2], do_print=True)

    assert config_1 == expected_generic_default_config_dict | {"nested_list": []}


def test_config_update_default_config():
    """Test updating the default config with itself as a sanity check on that config."""

    config_base_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "alphadia",
        "constants",
        "default.yaml",
    )
    config_1 = Config(name="default")
    config_1.from_yaml(config_base_path)

    config_2 = Config(name="also_default")
    config_2.from_yaml(config_base_path)

    config_1.update([config_2], do_print=True)

    assert config_1 == config_2


def test_config_update_none_values():
    """Test updating configs with None values in both directions."""
    config_with_none = Config(
        {
            "none_to_int": None,
            "none_to_dict": None,
            "none_to_list": None,
            "int_to_none": 42,
            "str_to_none": "hello",
        },
        "default",
    )

    update_config = Config(
        {
            "none_to_int": 123,
            "none_to_dict": {"nested_key": "nested_value"},
            "none_to_list": [1, 2, 3],
            "int_to_none": None,
            "str_to_none": None,
        },
        "update",
    )

    config_with_none.update([update_config])

    assert config_with_none == {
        "none_to_int": 123,
        "none_to_dict": {"nested_key": "nested_value"},
        "none_to_list": [1, 2, 3],
        "int_to_none": None,
        "str_to_none": None,
    }


def test_config_to_yaml_converts_numpy_types():
    """Test that numpy types are converted to native Python types when saving to YAML."""
    # given
    config_with_numpy = Config(
        {
            "numpy_float64": np.float64(5.0),
            "numpy_int32": np.int32(42),
            "numpy_array": np.array([1.0, 2.0, 3.0]),
            "nested": {
                "numpy_float32": np.float32(3.14),
                "list_with_numpy": [np.int64(100), np.float64(200.5)],
            },
        },
        "test",
    )

    # when
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        config_with_numpy.to_yaml(temp_path)

        with open(temp_path) as f2:
            yaml_content = f2.read()

        loaded_config = yaml.safe_load(yaml_content)

        # then
        assert isinstance(loaded_config["numpy_float64"], float)
        assert loaded_config["numpy_float64"] == 5.0

        assert isinstance(loaded_config["numpy_int32"], int)
        assert loaded_config["numpy_int32"] == 42

        assert isinstance(loaded_config["numpy_array"], list)
        assert loaded_config["numpy_array"] == [1.0, 2.0, 3.0]

        assert isinstance(loaded_config["nested"]["numpy_float32"], float)
        assert abs(loaded_config["nested"]["numpy_float32"] - 3.14) < 0.001

        assert isinstance(loaded_config["nested"]["list_with_numpy"][0], int)
        assert loaded_config["nested"]["list_with_numpy"][0] == 100

        assert isinstance(loaded_config["nested"]["list_with_numpy"][1], float)
        assert loaded_config["nested"]["list_with_numpy"][1] == 200.5

        assert "!!python/object" not in yaml_content
    finally:
        os.unlink(temp_path)
