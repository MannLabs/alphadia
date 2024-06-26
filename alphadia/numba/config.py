# native imports

# alphadia imports
# alpha family imports
# third party imports
import numpy as np

from alphadia.workflow import reporting


class JITConfig:
    """
    Base class for creating numba compatible config objects.

    Example
    -------

    Defining a config object tends to be verbose due to the strict typing requirements of numba.
    It requires to define a numba jitclass with the correct types for each attribute and a __init__ method.

    .. code-block:: python

        @nb.experimental.jitclass()
        class HybridCandidateConfigJIT():
            rt_tolerance: nb.float64
            def __init__(
                    self,
                    rt_tolerance,
                ):
                self.rt_tolerance = rt_tolerance


    For defining default values and updating the config object, the JITConfig class is used as base class.
    The config object must contain a class attribute `jit_container` pointing to the matching numba jitclass.
    The config object must implement a __init__ method where all parameters are initialized in the same order as they appear in the numba constructor.

    .. code-block:: python

        class HybridCandidateConfig(JITConfig):
            jit_container = HybridCandidateConfigJIT
            def __init__(self):
                self.rt_tolerance = 0.5

        config = HybridCandidateConfig()


    Default paramters can also be passed during construction.

    .. code-block:: python

        class HybridCandidateConfig(JITConfig):
            jit_container = HybridCandidateConfigJIT
            def __init__(self, rt_tolerance = 0.5):
                self.rt_tolerance = rt_tolerance

        config = HybridCandidateConfig(rt_tolerance = 0.1)

    The jit config can then retrived by calling the `jitclass` method.

    .. code-block:: python

        config = HybridCandidateConfig()
        jit_config = config.jitclass()
        print(jit_config.rt_tolerance)
        >> 0.5

    """

    def __init__(self):
        """Base class for creating numba compatible config objects.
        Note that this class is not meant to be instantiated. Classes inheriting from JITConfig must implement their own __init__ method.
        """
        self.reporter = reporting.Pipeline(
            backends=[
                reporting.LogBackend(),
            ]
        )
        raise NotImplementedError(
            "JITConfig is not meant to be instantiated. Classes inheriting from JITConfig must implement their own __init__ method."
        )

    def jitclass(self):
        """Returns a numba jitclass object.

        Returns
        -------

        jitclass : numba.experimental.jitclass.boxing.XXX
            Numba jitclass object with the type as defined in the jit_container class attribute.
        """

        self.validate()

        return self.jit_container(*self.__dict__.values())

    def validate(self):
        """Validates the config object.
        Note that this class is not meant to be instantiated. Classes inheriting from JITConfig must implement their own validate method.
        """

        raise NotImplementedError(
            "JITConfig is not meant to be instantiated. Classes inheriting from JITConfig must implement their own validate method."
        )

    def update(self, dict: dict):
        """Updates the config object with a dictionary of parameters.
        Can be used to update config object safely with a dictionary of parameters.

        Parameters
        ----------

        dict : dict
            Dictionary of parameters to update the config object.

        Example
        -------

        .. code-block:: python

            class HybridCandidateConfig(JITConfig):
                jit_container = HybridCandidateConfigJIT
                def __init__(self):
                    self.rt_tolerance = 0.5

            config = HybridCandidateConfig()
            config.update({'rt_tolerance': 0.1})

        """
        for key, value in dict.items():
            # check if attribute exists
            if not hasattr(self, key):
                self.reporter.log_string(
                    f"Parameter {key} does not exist in HybridCandidateConfig",
                    verbosity="error",
                )
                continue

            # check if types match
            if not isinstance(value, type(getattr(self, key))):
                try:
                    value = type(getattr(self, key))(value)
                except Exception:
                    self.reporter.log_string(
                        f"Parameter {key} has wrong type {type(value)}",
                        verbosity="error",
                    )

            # check if dtype matches
            if (
                isinstance(value, np.ndarray)
                and value.dtype != getattr(self, key).dtype
            ):
                try:
                    value = value.astype(getattr(self, key).dtype)
                except Exception:
                    self.reporter.log_string(
                        f"Parameter {key} has wrong dtype {value.dtype}",
                        verbosity="error",
                    )
                    continue

            # check if dimensions match
            if (
                isinstance(value, np.ndarray)
                and value.shape != getattr(self, key).shape
            ):
                self.reporter.log_string(
                    f"Parameter {key} has wrong shape {value.shape}",
                    verbosity="error",
                )
                continue

            # update attribute
            setattr(self, key, value)

    def __repr__(self) -> str:
        repr = f"<{self.__class__.__name__}, \n"
        for key, value in self.__dict__.items():
            repr += f"{key}={value} \n"

        repr += ">"
        return repr
