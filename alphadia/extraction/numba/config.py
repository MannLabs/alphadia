import numba as nb
import numpy as np
import logging
logger = logging.getLogger()
if not 'progress' in dir(logger):
    from alphadia.extraction import processlogger
    processlogger.init_logging()

class JITConfig():

    """
    Base class for creating numba compatible config objects.

    Example
    -------

    Defining a config object tends to be verbose due to the strict typing requirements of numba.
    It requires to define a numba jitclass with the correct types for each attribute and a __init__ method.
    
    ```python

    @nb.experimental.jitclass()
    class HybridCandidateConfigJIT():

        rt_tolerance: nb.float64

        def __init__(
                self, 
                rt_tolerance,
            ):

            self.rt_tolerance = rt_tolerance

    ```

    For defining default values and updating the config object, the JITConfig class is used as base class.
    The config object must contain a class attribute `jit_container` pointing to the matching numba jitclass.
    The config object must implement a __init__ method where all parameters are initialized in the same order as they appear in the numba constructor.

    ```python

    class HybridCandidateConfig(JITConfig):
    
        jit_container = HybridCandidateConfigJIT

        def __init__(self):
            
            self.rt_tolerance = 0.5

    config = HybridCandidateConfig()

    ```

    Default paramters can also be passed during construction.

    ```python

    class HybridCandidateConfig(JITConfig):
    
        jit_container = HybridCandidateConfigJIT

        def __init__(self, rt_tolerance = 0.5):
            
            self.rt_tolerance = rt_tolerance

    config = HybridCandidateConfig(rt_tolerance = 0.1)

    ```

    The jit config can then retrived by calling the `jitclass` method.

    ```python

    config = HybridCandidateConfig()
    jit_config = config.jitclass()
    print(jit_config.rt_tolerance)
    >> 0.5

    ```
    """

    def __init__(self):
        raise NotImplementedError('JITConfig is not meant to be instantiated. Classes inheriting from JITConfig must implement their own __init__ method.')
    
    def jitclass(self):

        return self.jit_container(
            *self.__dict__.values()
        )

    def update(self, dict):
        for key, value in dict.items():

            # check if attribute exists
            if not hasattr(self, key):
                logger.error(f'Parameter {key} does not exist in HybridCandidateConfig')
                continue

            # check if types match
            if not isinstance(value, type(getattr(self, key))):
                try:
                    value = type(getattr(self, key))(value)
                except:
                    logger.error(f'Parameter {key} has wrong type {type(value)}')

            # check if dtype matches
            if isinstance(value, np.ndarray):
                if value.dtype != getattr(self, key).dtype:
                    try:
                        value = value.astype(getattr(self, key).dtype)
                    except:
                        logger.error(f'Parameter {key} has wrong dtype {value.dtype}')
                        continue
            
            # check if dimensions match
            if isinstance(value, np.ndarray):
                if value.shape != getattr(self, key).shape:
                    logger.error(f'Parameter {key} has wrong shape {value.shape}')
                    continue

            # update attribute
            setattr(self, key, value)

    def __repr__(self) -> str:
        repr = ""
        for key, value in self.__dict__.items():
            repr += f'{key}: {value} \n'
        return repr