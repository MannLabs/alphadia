"""Abstract classes for the library transformations."""

import logging
import typing

logger = logging.getLogger()


class ProcessingStep:
    def __init__(self) -> None:
        """Base class for processing steps. Each implementation must implement the `validate` and `forward` method.
        Processing steps can be chained together in a ProcessingPipeline.
        """

    def __call__(self, input_obj: typing.Any) -> typing.Any:
        """Run the processing step on the input object."""
        logger.info(f"Running {self.__class__.__name__}")
        if self.validate(input_obj):
            return self.forward(input_obj)
        logger.critical(
            f"Input {input_obj} failed validation for {self.__class__.__name__}"
        )
        raise ValueError(
            f"Input {input_obj} failed validation for {self.__class__.__name__}"
        )

    def validate(self, input_obj: typing.Any) -> bool:
        """Validate the input object."""
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, input_obj: typing.Any) -> typing.Any:
        """Run the processing step on the input object."""
        raise NotImplementedError("Subclasses must implement this method")


class ProcessingPipeline:
    def __init__(self, steps: list[ProcessingStep]) -> None:
        """Processing pipeline for loading and transforming spectral libraries.

        The pipeline is a list of ProcessingStep objects. Each step is called in order
        and the output of the previous step is passed to the next step.

        Example::

            pipeline = ProcessingPipeline([
                DynamicLoader(),
                PrecursorInitializer(),
                AnnotateFasta(fasta_path_list),
                IsotopeGenerator(),
                DecoyGenerator(),
                RTNormalization()
            ])

            library = pipeline(input_path)

        """
        self.steps = steps

    def __call__(self, input_obj: typing.Any) -> typing.Any:
        """Run the pipeline on the input object."""
        for step in self.steps:
            input_obj = step(input_obj)
        return input_obj
