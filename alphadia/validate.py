# native imports
import logging

import numpy as np

# alphadia imports
# alpha family imports
# third party imports
import pandas as pd

logger = logging.getLogger()


class Property:
    """Column property base class"""

    def __init__(self, name, type):
        """
        Base class for all properties

        Parameters
        ----------

        name: str
            Name of the property

        type: type
            Type of the property

        """
        self.name = name
        self.type = type


class Optional(Property):
    """Optional property"""

    def __init__(self, name, type):
        """
        Optional property

        Parameters
        ----------

        name: str
            Name of the property

        type: type
            Type of the property

        """

        self.name = name
        self.type = type

    def __call__(self, df, logging=True):
        """
        Casts the property to the specified type if it is present in the dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe to validate

        logging: bool
            If True, log the validation results
        """

        if self.name in df.columns and df[self.name].dtype != self.type:
            df[self.name] = df[self.name].astype(self.type)

        return True


class Required(Property):
    """Required property"""

    def __init__(self, name, type):
        """
        Required property

        Parameters
        ----------

        name: str
            Name of the property

        type: type
            Type of the property

        """
        self.name = name
        self.type = type

    def __call__(self, df, logging=True):
        """
        Casts the property to the specified type if it is present in the dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe to validate

        logging: bool
            If True, log the validation results

        """

        if self.name in df.columns:
            if df[self.name].dtype != self.type:
                df[self.name] = df[self.name].astype(self.type)

            return True
        else:
            return False


class Schema:
    def __init__(self, name, properties):
        """
        Schema for validating dataframes

        Parameters
        ----------

        name: str
            Name of the schema

        properties: list
            List of Property objects

        """

        self.name = name
        self.schema = properties
        for property in self.schema:
            if not isinstance(property, Property):
                raise ValueError("Schema must contain only Property objects")

    def __call__(self, df, logging=True):
        """
        Validates the dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe to validate

        logging: bool
            If True, log the validation results

        """

        for property in self.schema:
            if not property(df, logging=logging):
                raise ValueError(
                    f"Validation of {self.name} failed: Column {property.name} is not present in the dataframe"
                )

    def docstring(self) -> str:
        """Automatically generate a docstring for the schema.

        Returns
        -------
        str
            Docstring for the schema
        """

        docstring = """
    Schema
    ------

    .. list-table::
        :widths: 1 1 1
        :header-rows: 1

        * - Name
          - Required
          - Type
"""
        for property in self.schema:
            emphasis = "**" if isinstance(property, Required) else ""
            docstring += f"""
        * - {property.name}
          - {emphasis}{property.__class__.__name__}{emphasis}
          - {property.type.__name__}
        """
        return docstring


precursors_flat_schema = Schema(
    "precursors_flat",
    [
        Required("elution_group_idx", np.uint32),
        Optional("score_group_idx", np.uint32),
        Required("precursor_idx", np.uint32),
        Required("channel", np.uint32),
        Required("decoy", np.uint8),
        Required("flat_frag_start_idx", np.uint32),
        Required("flat_frag_stop_idx", np.uint32),
        Required("charge", np.uint8),
        Required("rt_library", np.float32),
        Optional("rt_calibrated", np.float32),
        Required("mobility_library", np.float32),
        Optional("mobility_calibrated", np.float32),
        Required("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        Required("proteins", object),
        Required("genes", object),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


def precursors_flat(df: pd.DataFrame, logging: bool = True):
    """Validate flat precursor dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Precursor dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------

    ValueError
        If validation fails

    """
    check_critical_values(df)
    precursors_flat_schema(df, logging=logging)


precursors_flat.__doc__ += precursors_flat_schema.docstring()

fragments_flat_schema = Schema(
    "fragments_flat",
    [
        Required("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        Required("intensity", np.float32),
        Required("cardinality", np.uint8),
        Required("type", np.uint8),
        Required("loss_type", np.uint8),
        Required("charge", np.uint8),
        Required("number", np.uint8),
        Required("position", np.uint8),
    ],
)


def fragments_flat(df: pd.DataFrame, logging: bool = True):
    """Validate flat fragment dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Fragment dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------

    ValueError
        If validation fails

    """
    check_critical_values(df)
    fragments_flat_schema(df, logging=logging)


fragments_flat.__doc__ += fragments_flat_schema.docstring()

candidates_schema = Schema(
    "candidates_df",
    [
        Required("elution_group_idx", np.uint32),
        Required("precursor_idx", np.uint32),
        Required("rank", np.uint8),
        Required("scan_start", np.int64),
        Required("scan_stop", np.int64),
        Required("scan_center", np.int64),
        Required("frame_start", np.int64),
        Required("frame_stop", np.int64),
        Required("frame_center", np.int64),
        Optional("score", np.float32),
        Optional("score_group_idx", np.uint32),
        Optional("channel", np.uint8),
        Optional("decoy", np.uint8),
        Optional("flat_frag_start_idx", np.uint32),
        Optional("flat_frag_stop_idx", np.uint32),
        Optional("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


def candidates_df(df: pd.DataFrame, logging: bool = True):
    """Validate candidate dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Candidate dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------

    ValueError
        If validation fails

    """
    check_critical_values(df)
    candidates_schema(df, logging=logging)


candidates_df.__doc__ += candidates_schema.docstring()


def check_critical_values(input_df):
    for col in input_df.columns:
        if np.issubdtype(input_df[col].dtype, np.floating):
            nan_count = input_df[col].isna().sum()
            inf_count = np.isinf(input_df[col]).sum()

            if nan_count > 0:
                nan_percentage = nan_count / len(input_df) * 100
                logger.warning(
                    f"{col} has {nan_count} NaNs ( {nan_percentage:.2f} % out of {len(input_df)})"
                )

            if inf_count > 0:
                inf_percentage = inf_count / len(input_df) * 100
                logger.warning(
                    f"{col} has {inf_count} Infs ( {inf_percentage:.2f} % out of {len(input_df)})"
                )


features_schema = Schema(
    "candidate_features_df",
    [
        Required("precursor_idx", np.uint32),
        Required("elution_group_idx", np.uint32),
        Required("rank", np.uint8),
        Required("decoy", np.uint8),
        Required("channel", np.uint8),
        Required("charge", np.uint8),
        Required("flat_frag_start_idx", np.uint32),
        Required("flat_frag_stop_idx", np.uint32),
        Required("scan_center", np.int64),
        Required("scan_start", np.int64),
        Required("scan_stop", np.int64),
        Required("frame_center", np.int64),
        Required("frame_start", np.int64),
        Required("frame_stop", np.int64),
        Required("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        Required("mz_observed", np.float32),
        Required("rt_library", np.float32),
        Optional("rt_calibrated", np.float32),
        Required("rt_observed", np.float32),
        Required("mobility_library", np.float32),
        Optional("mobility_calibrated", np.float32),
        Required("mobility_observed", np.float32),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


def candidate_features_df(input_df: pd.DataFrame, logging: bool = True):
    """Validate feature dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------

    ValueError
        If validation fails

    """
    check_critical_values(input_df)
    features_schema(input_df, logging=logging)


candidate_features_df.__doc__ += features_schema.docstring()

fragment_features_schema = Schema(
    "fragment_features_df",
    [
        Required("precursor_idx", np.uint32),
        Required("rank", np.uint8),
        Required("elution_group_idx", np.uint32),
        Required("mz_library", np.float32),
        Required("mz_observed", np.float32),
        Required("mass_error", np.float32),
        Required("height", np.float32),
        Required("intensity", np.float32),
        Required("decoy", np.uint8),
    ],
)


def fragment_features_df(input_df: pd.DataFrame, logging: bool = True):
    """Validate fragment feature dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Fragment feature dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------

    ValueError
        If validation fails

    """
    check_critical_values(input_df)
    fragment_features_schema(input_df, logging=logging)


fragment_features_df.__doc__ += fragment_features_schema.docstring()
