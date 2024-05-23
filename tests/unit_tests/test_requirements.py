# 169:import os
import re

import pytest

from packaging.requirements import Requirement

# special comment to tolerate version in loose requirements file
TOLERATE_VERSION_COMMENT = "test:tolerate_version"


def _split_at_first_hash(input_string: str) -> tuple[str, ...]:
    """Split input string at the first occurrence of '#'.

    Always returns a tuple of two strings, even if the input string does not contain a '#'.
    """

    # (?<!\\) is a negative lookbehind assertion that ensures the # is not preceded by a backslash
    # (escaping the # would prevent the split at that point).
    parts = re.split(r"(?<!\\)#", input_string, 1)
    if len(parts) == 1:
        parts.append("")
    return tuple([p.strip() for p in parts])


def parse_requirements(file_path: str) -> dict[str, tuple[Requirement, str]]:
    """
    Parse a requirements file and return a dictionary of packages with their comments.

    Parameters
    ----------
    file_path : str
        The path to the requirements file to parse.

    Returns
    -------
    dict:
        A dictionary of packages with their comments.
        The keys are the package names, and the values are tuples of the form (Requirement, str).
        The str is the comment associated with the package in the requirements file.

    """
    packages = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                req_string, comment = _split_at_first_hash(line)

                req = Requirement(req_string)
                if req.name in packages:
                    raise ValueError(
                        f"Duplicate package '{req.name}' found in requirements file"
                    )

                packages[req.name] = (req, comment)

    return packages


def test_requirements():
    """Test the strict and loose requirements.

    The strict requirements must have one fixed version.

    All requirements must be present in the loose requirements.
    The loose requirements should not have a fixed version unless an exception is
    stated by the "test:tolerate_version" comment.
    """

    requirements_path = "../../requirements/"
    file_name_strict = "requirements.txt"
    file_name_loose = "requirements_loose.txt"
    file_path_strict = os.path.join(requirements_path, file_name_strict)
    file_path_loose = os.path.join(requirements_path, file_name_loose)

    reqs_strict = parse_requirements(file_path_strict)
    reqs_loose = parse_requirements(file_path_loose)

    req_loose_names = reqs_loose.keys()
    req_strict_names = reqs_strict.keys()

    set_loose = set(req_loose_names)
    set_strict = set(req_strict_names)
    assert (
        set_strict == set_loose
    ), f"Requirements in do not match. only in strict: {set_strict-set_loose}; only in loose: {set_loose-set_strict}"

    for req, _ in reqs_strict.values():
        assert (
            len(req.specifier) == 1
        ), f"Requirement '{req}' does not have one defined version in {file_name_strict}"
        assert str(
            list(req.specifier)[0]
        ).startswith(
            "=="
        ), f"Requirement '{req}' does not have a fixed version ('==') in {file_name_strict}"

    for req_name, (req, comment) in reqs_loose.items():
        if comment != TOLERATE_VERSION_COMMENT:
            assert (
                len(req.specifier) == 0
            ), f"Requirement '{req}' must not have a defined version in {file_name_loose}"
        else:
            # here we rely on the test for 'fixed version' above to access the specifier
            specifier_strict = reqs_strict[req_name][0].specifier
            version_strict = str(list(specifier_strict)[0]).replace("==", "")
            specifier_loose = req.specifier
            assert specifier_loose.contains(
                version_strict
            ), f"Requirement '{req}' is too strict in {file_name_loose}"
