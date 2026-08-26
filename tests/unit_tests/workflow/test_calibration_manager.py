"""Unit tests for the CalibrationManager class."""

from alphadia.workflow.managers.calibration_manager import (
    CalibrationEstimators,
    CalibrationGroups,
    CalibrationManager,
)


def test_estimator_groups_use_allowed_names() -> None:
    """The manager builds estimators using only allowed group/estimator names."""

    allowed_groups = CalibrationGroups.get_values()
    allowed_estimators = CalibrationEstimators.get_values()

    manager = CalibrationManager(load_from_file=False)

    errors = []
    for group_name, group in manager.estimator_groups.items():
        if group_name not in allowed_groups:
            errors.append(
                f"Invalid calibration group '{group_name}'. Allowed groups are: {allowed_groups}"
            )
        for estimator_name in group:
            if estimator_name not in allowed_estimators:
                errors.append(
                    f"Invalid estimator '{estimator_name}' in group '{group_name}'. Allowed estimators are: {allowed_estimators}"
                )
    if errors:
        raise AssertionError("Invalid calibration configuration:\n" + "\n".join(errors))


def test_estimator_groups_gating() -> None:
    """MS1 and mobility estimators are skipped when not available in the raw data."""

    manager = CalibrationManager(
        load_from_file=False, has_ms1=False, has_mobility=False
    )

    precursor = manager.estimator_groups[CalibrationGroups.PRECURSOR]
    assert CalibrationEstimators.MZ not in precursor
    assert CalibrationEstimators.MOBILITY not in precursor
    assert CalibrationEstimators.RT in precursor

    manager_full = CalibrationManager(load_from_file=False)
    precursor_full = manager_full.estimator_groups[CalibrationGroups.PRECURSOR]
    assert CalibrationEstimators.MZ in precursor_full
    assert CalibrationEstimators.MOBILITY in precursor_full
    assert CalibrationEstimators.RT in precursor_full
