"""Tests for model train."""

import pytest

from aiida.common import InputValidationError, datastructures
from aiida.engine import run
from aiida.orm import Bool
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData

# this is just a temporary solution till mace gets a tag on current main.
try:
    from mace.cli.run_train import run as run_train  # pylint: disable=unused-import

    MACE_IMPORT_ERROR = False
except ImportError:
    MACE_IMPORT_ERROR = True


def test_prepare_train(fixture_sandbox, generate_calc_job, janus_code, config_folder):
    """Test generating singlepoint calculation job."""

    entry_point_name = "janus.train"
    config_path = config_folder / "mlip_train.yml"
    config = JanusConfigfile(file=config_path)
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "mlip_config": config,
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    retrieve_list = [
        calc_info.uuid,
        "aiida-stdout.txt",
        "logs",
        "results",
        "checkpoints",
        "test.model",
        "test_compiled.model",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert fixture_sandbox.get_content_list() == ["mlip_train.yml"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_file_error(
    fixture_sandbox, generate_calc_job, janus_code, config_folder, tmp_path
):
    """Test error if path for xyz is non existent."""

    entry_point_name = "janus.train"
    config_path = config_folder / "mlip_train.yml"

    # Temporarily modify config file to introduce an error
    with open(config_path, encoding="utf-8") as file:
        right_path = file.read()

    wrong_path = right_path.replace("mlip_train.xyz", "mlip_train_wrong.xyz")
    with open(tmp_path / "mlip_config.yml", "w", encoding="utf-8") as file:
        file.write(wrong_path)

    config = JanusConfigfile(file=tmp_path / "mlip_config.yml")
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "mlip_config": config,
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_noname(
    fixture_sandbox, generate_calc_job, janus_code, config_folder, tmp_path
):
    """Test error if no 'name' keyword is given in config."""

    entry_point_name = "janus.train"
    config_path = config_folder / "mlip_train.yml"

    # Temporarily modify config file to introduce an error
    with open(config_path, encoding="utf-8") as file:
        original_lines = file.readlines()

    noname_lines = [line for line in original_lines if "name" not in line]

    with open(tmp_path / "mlip_config.yml", "w", encoding="utf-8") as file:
        file.writelines(noname_lines)

    config = JanusConfigfile(file=tmp_path / "mlip_config.yml")
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "mlip_config": config,
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    # Restore config file
    with open(config_path, "w", encoding="utf-8") as file:
        file.writelines(original_lines)


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
        "foundation_model": ModelData.local_file(
            file=model_file, architecture="mace_mp"
        ),
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


@pytest.mark.skipif(MACE_IMPORT_ERROR, reason="Requires updated version of MACE")
def test_run_train(janus_code, config_folder):
    """Test running train with fine-tuning calculation"""

    model_file = config_folder / "test.model"
    config_path = config_folder / "mlip_train.yml"
    config = JanusConfigfile(file=config_path)
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "fine_tune": Bool(True),
        "code": janus_code,
        "mlip_config": config,
        "foundation_model": ModelData.local_file(
            file=model_file, architecture="mace_mp"
        ),
    }

    trainfinetuneCalc = CalculationFactory("janus.train")
    result = run(trainfinetuneCalc, **inputs)

    assert "results_dict" in result
    obtained_res = result["results_dict"].get_dict()
    assert "logs" in result
    assert obtained_res["loss"] == pytest.approx(0.062798671424389)
