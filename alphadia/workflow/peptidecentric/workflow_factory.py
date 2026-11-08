"""Selection of the peptide-centric workflow implementation by extraction backend."""

from alphadia.constants.keys import ExtractionBackend
from alphadia.exceptions import ConfigError
from alphadia.workflow.config import Config
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow
from alphadia.workflow.peptidecentric.python.workflow import (
    PythonPeptideCentricWorkflow,
)
from alphadia.workflow.peptidecentric.rust.workflow import RustPeptideCentricWorkflow

_WORKFLOW_CLASSES: dict[str, type[PeptideCentricWorkflow]] = {
    ExtractionBackend.PYTHON: PythonPeptideCentricWorkflow,
    ExtractionBackend.RUST: RustPeptideCentricWorkflow,
}


def get_extraction_backend(config: Config) -> str:
    """Return the configured `search.extraction_backend`, normalized to one of `ExtractionBackend`."""
    return config["search"]["extraction_backend"].lower()


def get_workflow_class(config: Config) -> type[PeptideCentricWorkflow]:
    """Return the workflow class for the configured `search.extraction_backend`.

    Raises
    ------
    ConfigError
        If the extraction backend is not supported.
    """
    backend = get_extraction_backend(config)

    if backend not in _WORKFLOW_CLASSES:
        raise ConfigError(
            "search.extraction_backend",
            backend,
            "final",
            f"Supported backends are: {ExtractionBackend.get_values()}",
        )

    return _WORKFLOW_CLASSES[backend]
