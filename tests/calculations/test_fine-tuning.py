"""Tests for model train."""

import pytest

from aiida.common import InputValidationError, datastructures
from aiida.orm import Bool

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData


def test_prepare_tune(fixture_sandbox, generate_calc_job, janus_code, config_folder):
    """Test generating fine tuning calculation job."""

    model_file = config_folder / "test.model"
    entry_point_name = "janus.train"
    config_path = config_folder / "mlip_train.yml"
    config = JanusConfigfile(file=config_path)
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "mlip_config": config,
        "fine_tune": Bool(True),
        "model": ModelData.local_file(file=model_file, architecture="mace_mp"),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = ["train", "--mlip-config", "mlip_train.yml", "--fine-tune"]

    retrieve_list = [
        calc_info.uuid,
        "aiida-stdout.txt",
        "logs",
        "results",
        "checkpoints",
        "test.model",
        "test_compiled.model",
    ]
    print(calc_info.codes_info[0].cmdline_params)
    print(cmdline_params)
    # Check the attributes of the returned `CalcInfo`
    assert fixture_sandbox.get_content_list() == ["mlip_train.yml"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    assert calc_info.codes_info[0].cmdline_params == cmdline_params


def test_finetune_error(fixture_sandbox, generate_calc_job, janus_code, config_folder):
    """Test error if no model is given."""

    entry_point_name = "janus.train"
    config_path = config_folder / "mlip_train.yml"
    config = JanusConfigfile(file=config_path)
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "fine_tune": Bool(True),
        "code": janus_code,
        "mlip_config": config,
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)
