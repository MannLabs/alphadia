"""Selection of the peptide-centric workflow implementation by extraction backend."""

from alphadia.constants.keys import ExtractionBackend
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


def get_workflow_class(config: Config) -> type[PeptideCentricWorkflow]:
    """Return the workflow class for the configured `search.extraction_backend`.

    Raises
    ------
    ValueError
        If the extraction backend is not supported.
    """
    backend = config["search"]["extraction_backend"].lower()

    if backend not in _WORKFLOW_CLASSES:
        raise ValueError(
            f"Invalid extraction backend '{backend}'. "
            f"Supported backends are: {ExtractionBackend.get_values()}"
        )

    return _WORKFLOW_CLASSES[backend]
