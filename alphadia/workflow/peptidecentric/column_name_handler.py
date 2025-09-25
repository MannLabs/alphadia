from alphadia.constants.keys import (
    CalibCols,
)
from alphadia.workflow.managers.calibration_manager import (
    CalibrationEstimators,
    CalibrationGroups,
    CalibrationManager,
)


class ColumnNameHandler:
    """A class to handle column names in a peptide-centric workflow.

    The class determines the appropriate column names for precursor and fragment m/z, retention time (RT), and mobility based on whether calibration has been performed and the presence of MS1 and mobility data.
    """

    def __init__(
        self,
        calibration_manager: CalibrationManager,
        *,
        dia_data_has_ms1: bool,
        dia_data_has_mobility: bool,
    ) -> None:
        """Initializes the ColumnNameHandler."""
        self._estimator_groups = calibration_manager.estimator_groups
        self._dia_data_has_ms1 = dia_data_has_ms1
        self._dia_data_has_mobility = dia_data_has_mobility

    def get_precursor_mz_column(self) -> str:
        """Get the precursor m/z column name.

        This function will return CalibCols.MZ_CALIBRATED if precursor MZ calibration has happened, otherwise it will return CalibCols.MZ_LIBRARY.
        If no MS1 data is present, it will always return CalibCols.MZ_LIBRARY.

        Returns
        -------
        str
            Name of the precursor m/z column
        """
        if (
            self._dia_data_has_ms1
            and self._estimator_groups[CalibrationGroups.PRECURSOR][
                CalibrationEstimators.MZ
            ].is_fitted
        ):
            return CalibCols.MZ_CALIBRATED

        return CalibCols.MZ_LIBRARY

    def get_fragment_mz_column(self) -> str:
        """Get the fragment m/z column name.

        This function will return CalibCols.MZ_CALIBRATED if fragment MZ calibration has happened, otherwise it will return CalibCols.MZ_LIBRARY.

        Returns
        -------
        str
            Name of the fragment m/z column
        """
        if self._estimator_groups[CalibrationGroups.FRAGMENT][
            CalibrationEstimators.MZ
        ].is_fitted:
            return CalibCols.MZ_CALIBRATED

        return CalibCols.MZ_LIBRARY

    def get_rt_column(self) -> str:
        """Get the precursor rt column name.

        This function will return CalibCols.RT_CALIBRATED if precursor RT calibration has happened, otherwise it will return CalibCols.RT_LIBRARY.
        If no MS1 data is present, it will always return CalibCols.RT_LIBRARY.

        Returns
        -------
        str
            Name of the precursor rt column
        """
        if self._estimator_groups[CalibrationGroups.PRECURSOR][
            CalibrationEstimators.RT
        ].is_fitted:
            return CalibCols.RT_CALIBRATED

        return CalibCols.RT_LIBRARY

    def get_mobility_column(self) -> str:
        """Get the precursor mobility column name.

        This function will return CalibCols.MOBILITY_CALIBRATED if precursor mobility calibration has happened, otherwise it will return CalibCols.MOBILITY_LIBRARY.
        If no mobility data is present, it will always return CalibCols.MOBILITY_LIBRARY.

        Returns
        -------
        str
            Name of the precursor mobility column
        """
        if (
            self._dia_data_has_mobility
            and self._estimator_groups[CalibrationGroups.PRECURSOR][
                CalibrationEstimators.MOBILITY
            ].is_fitted
        ):
            return CalibCols.MOBILITY_CALIBRATED

        return CalibCols.MOBILITY_LIBRARY
