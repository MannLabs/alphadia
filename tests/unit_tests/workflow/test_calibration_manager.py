"""Unit tests for the CalibrationManager class."""

from alphadia.workflow.managers.calibration_manager import (
    CALIBRATION_GROUPS_CONFIG,
    CalibrationEstimators,
    CalibrationGroups,
)


def test_validate_calibration_config() -> None:
    """Validate the calibration configuration is using only allowed keys."""

    allowed_groups = CalibrationGroups.get_values()
    allowed_estimators = CalibrationEstimators.get_values()

    errors = []
    for group_name, group in CALIBRATION_GROUPS_CONFIG.items():
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
