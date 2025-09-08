from alphadia.constants.keys import (
    COLUMN_TYPE_CALIBRATED,
    COLUMN_TYPE_LIBRARY,
    CalibCols,
)
from alphadia.workflow.managers.optimization_manager import OptimizationManager


class ColumnNameHandler:
    """
    A class to handle column names in a peptide-centric workflow.
    """

    def __init__(
        self,
        optimization_manager: OptimizationManager,
        *,
        dia_data_has_ms1: bool,
        dia_data_has_mobility: bool,
    ) -> None:
        """
        Initializes the ColumnNameHandler.
        """
        self._optimization_manager = optimization_manager
        self._dia_data_has_ms1 = dia_data_has_ms1
        self._dia_data_has_mobility = dia_data_has_mobility

    def get_precursor_mz_column(self) -> str:
        """Get the precursor m/z column name.

        This function will return MRMCols.MZ_CALIBRATED if precursor calibration has happened, otherwise it will return MRMCols.MZ_LIBRARY.
        If no MS1 data is present, it will always return MRMCols.MZ_LIBRARY.

        Returns
        -------
        str
            Name of the precursor m/z column
        """
        if (
            not self._dia_data_has_ms1
            or self._optimization_manager.column_type == COLUMN_TYPE_LIBRARY
        ):
            return CalibCols.MZ_LIBRARY

        if self._optimization_manager.column_type == COLUMN_TYPE_CALIBRATED:
            return CalibCols.MZ_CALIBRATED

        raise ValueError(
            f"Unknown column type: {self._optimization_manager.column_type}"
        )

    def get_fragment_mz_column(self) -> str:
        """Get the fragment m/z column name.

        This function will return MRMCols.MZ_CALIBRATED if fragment calibration has happened, otherwise it will return MRMCols.MZ_LIBRARY.

        Returns
        -------
        str
            Name of the fragment m/z column
        """

        if self._optimization_manager.column_type == COLUMN_TYPE_LIBRARY:
            return CalibCols.MZ_LIBRARY

        if self._optimization_manager.column_type == COLUMN_TYPE_CALIBRATED:
            return CalibCols.MZ_CALIBRATED

        raise ValueError(
            f"Unknown column type: {self._optimization_manager.column_type}"
        )

    def get_rt_column(self) -> str:
        """Get the precursor rt column name.

        This function will return MRMCols.RT_CALIBRATED if precursor calibration has happened, otherwise it will return MRMCols.RT_LIBRARY.
        If no MS1 data is present, it will always return MRMCols.RT_LIBRARY.

        Returns
        -------
        str
            Name of the precursor rt column
        """
        if self._optimization_manager.column_type == COLUMN_TYPE_LIBRARY:
            return CalibCols.RT_LIBRARY

        if self._optimization_manager.column_type == COLUMN_TYPE_CALIBRATED:
            return CalibCols.RT_CALIBRATED

        raise ValueError(
            f"Unknown column type: {self._optimization_manager.column_type}"
        )

    def get_mobility_column(self) -> str:
        """Get the precursor mobility column name.

        This function will return MRMCols.MOBILITY_CALIBRATED if precursor calibration has happened, otherwise it will return MRMCols.MOBILITY_LIBRARY.
        If no mobility data is present, it will always return MRMCols.MOBILITY_LIBRARY.

        Returns
        -------
        str
            Name of the precursor mobility column
        """
        if (
            not self._dia_data_has_mobility
            or self._optimization_manager.column_type == COLUMN_TYPE_LIBRARY
        ):
            return CalibCols.MOBILITY_LIBRARY

        if self._optimization_manager.column_type == COLUMN_TYPE_CALIBRATED:
            return CalibCols.MOBILITY_CALIBRATED

        raise ValueError(
            f"Unknown column type: {self._optimization_manager.column_type}"
        )
