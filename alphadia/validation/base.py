# native imports
import logging

# alphadia imports
# alpha family imports
# third party imports

logger = logging.getLogger()


class Property:
    """Column property base class"""

    def __init__(self, name, type):
        """Base class for all properties

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
        """Optional property

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
        """Casts the property to the specified type if it is present in the dataframe

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
        """Required property

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
        """Casts the property to the specified type if it is present in the dataframe

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
        return False


class Schema:
    def __init__(self, name, properties):
        """Schema for validating dataframes

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
        """Validates the dataframe

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
