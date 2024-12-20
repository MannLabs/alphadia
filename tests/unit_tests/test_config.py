import os
from io import StringIO

import pytest
import yaml

from alphadia.workflow.config import Config

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
    config_1.update([config_2, config_3], do_print=True)

    config_1.__repr__()
    print("X")

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
    config_1.update([config_2], do_print=True)

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


def test_config_update_advanced_add_nested_list_item():
    """Test updating a config with nested values and lists, with new list longer than old one."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict(
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
        }
    )

    # when
    config_1.update([config_2], do_print=True)

    assert config_1.to_dict() == expected_generic_default_config_dict | {
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
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict({"new_key": 0})

    # when
    with pytest.raises(ValueError, match="Key not found in target_config: 'new_key'"):
        config_1.update([config_2], do_print=True)


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
        config_1.update([config_2], do_print=True)


def test_config_update_new_key_in_nested_list():
    """Test updating a config with a new item in a nested list."""
    config_1 = Config("default")
    config_1.from_dict(yaml.safe_load(StringIO(generic_default_config)))

    config_2 = Config("first")
    config_2.from_dict(
        {"nested_list": [{"key4": "value44", "name": "nested_list_value_3"}]}
    )

    # when
    config_1.update([config_2], do_print=True)

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
    config_1.update([config_2], do_print=True)

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

    config_1.update([config_2], do_print=True)

    assert config_1.to_dict() == config_2.to_dict()
