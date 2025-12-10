import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alphadia.workflow import base
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager


@pytest.mark.skip(reason="no input data given")  # TODO re-enable or delete
def test_workflow_base():
    if pytest.test_data is None:
        raise ValueError("No test data found")

    for _, file_list in pytest.test_data.items():
        for file in file_list:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "misc", "config", "default.yaml"
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["output_directory"] = tempfile.gettempdir()

            workflow_name = Path(file).stem

            my_workflow = base.WorkflowBase(
                workflow_name,
                config,
            )
            my_workflow.load(file, pd.DataFrame({}))

            assert my_workflow.config["output_directory"] == config["output_directory"]
            assert my_workflow.instance_name == workflow_name
            assert my_workflow.path == os.path.join(
                config["output_directory"], base.QUANT_FOLDER_NAME, workflow_name
            )

            assert os.path.exists(my_workflow.path)

            # assert isinstance(my_workflow.dia_data, bruker.TimsTOFTranspose) or isinstance(my_workflow.dia_data, thermo.Thermo)
            assert isinstance(my_workflow.calibration_manager, CalibrationManager)
            assert isinstance(my_workflow.optimization_manager, OptimizationManager)

            # os.rmdir(os.path.join(my_workflow.path, my_workflow.FIGURE_PATH))
            # os.rmdir(os.path.join(my_workflow.path))
            shutil.rmtree(os.path.join(my_workflow.path))
