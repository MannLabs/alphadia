from abc import ABC, abstractmethod

import numpy as np


class JITConfig(ABC):
    """Base class for creating numba compatible config objects.

    Example
    -------

    Defining a config object tends to be verbose due to the strict typing requirements of numba.
    It requires to define a numba jitclass with the correct types for each attribute and a __init__() method.

    .. code-block:: python

        @nb.experimental.jitclass()
        class CandidateSelectionConfigJIT():
            rt_tolerance: nb.float64
            def __init__(self, rt_tolerance):
                self.rt_tolerance = rt_tolerance


    For defining default values and updating the config object, the JITConfig class is used as base class.
    The config object must contain a class attribute `_jit_container_type` pointing to the matching numba jitclass.
    The config object must implement a __init__() method where all parameters are initialized in the same order as they appear in the numba constructor.

    .. code-block:: python

        class CandidateSelectionConfig(JITConfig):
            _jit_container_type = CandidateSelectionConfigJIT
            def __init__(self, rt_tolerance = 0.5):
                self.rt_tolerance = rt_tolerance

        config = CandidateSelectionConfig()


    The jit config can then be retrieved by calling the `to_jitclass()` method.

    .. code-block:: python

        config = CandidateSelectionConfig()
        jit_config = config.to_jitclass()
        print(jit_config.rt_tolerance)
        >> 0.5

    """

    _jit_container_type: type

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Base class for creating numba compatible config objects."""

    def to_jitclass(self):
        """Create a numba jitclass object with the current state of this class.

        Returns
        -------

        jitclass : numba.experimental.jitclass.boxing.XXX
            Numba jitclass object with the type as defined in the _jit_container_type class attribute.
        """

        self.validate()

        return self._jit_container_type(*self.__dict__.values())

    @abstractmethod
    def validate(self):
        """Validates the config object.
        Note that this class is not meant to be instantiated. Classes inheriting from JITConfig must implement their own validate method.
        """

    def update(self, input_dict: dict):
        """Updates the config object with a dictionary of parameters.

        Can be used to update config object safely with a dictionary of parameters.

        Parameters
        ----------

        input_dict : dict
            Dictionary of parameters to update the config object.

        Example
        -------

        .. code-block:: python

            class CandidateSelectionConfig(JITConfig):
                _jit_container_type = CandidateSelectionConfigJIT
                def __init__(self):
                    self.rt_tolerance = 0.5

            config = CandidateSelectionConfig()
            config.update({'rt_tolerance': 0.1})

        """
        for key, value in input_dict.items():
            # check if attribute exists
            if not hasattr(self, key):
                raise ValueError(
                    f"Parameter {key} does not exist in CandidateSelectionConfig",
                )

            # check if types match
            if not isinstance(value, type(getattr(self, key))):
                try:
                    value = type(getattr(self, key))(value)
                except Exception as e:
                    raise ValueError(
                        f"Parameter {key} has wrong type {type(value)}",
                    ) from e

            # check if dtype matches
            if (
                isinstance(value, np.ndarray)
                and value.dtype != getattr(self, key).dtype
            ):
                try:
                    value = value.astype(getattr(self, key).dtype)
                except Exception as e:
                    raise ValueError(
                        f"Parameter {key} has wrong dtype {value.dtype}",
                    ) from e

            # check if dimensions match
            if (
                isinstance(value, np.ndarray)
                and value.shape != getattr(self, key).shape
            ):
                raise ValueError(
                    f"Parameter {key} has wrong shape {value.shape}",
                )

            # update attribute
            setattr(self, key, value)

    def __repr__(self) -> str:
        repr = f"<{self.__class__.__name__}, \n"
        for key, value in self.__dict__.items():
            repr += f"{key}={value} \n"

        repr += ">"
        return repr
