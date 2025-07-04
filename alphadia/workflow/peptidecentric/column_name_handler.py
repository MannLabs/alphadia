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
        self.optimization_manager = optimization_manager
        self.dia_data_has_ms1 = dia_data_has_ms1
        self.dia_data_has_mobility = dia_data_has_mobility

    def get_precursor_mz_column(self):
        """Get the precursor m/z column name.

        This function will return `mz_calibrated` if precursor calibration has happened, otherwise it will return `mz_library`.
        If no MS1 data is present, it will always return `mz_library`.

        Returns
        -------
        str
            Name of the precursor m/z column

        """
        return (
            f"mz_{self.optimization_manager.column_type}"
            if self.dia_data_has_ms1
            else "mz_library"
        )

    def get_fragment_mz_column(self):
        return f"mz_{self.optimization_manager.column_type}"

    def get_rt_column(self):
        return f"rt_{self.optimization_manager.column_type}"

    def get_mobility_column(self):
        return (
            f"mobility_{self.optimization_manager.column_type}"
            if self.dia_data_has_mobility
            else "mobility_library"
        )
