"""Tests for model train."""

import pytest

from aiida.common import InputValidationError, datastructures

from aiida_mlip.data.config import JanusConfigfile


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

    print(sorted(calc_info.retrieve_list))
    print(retrieve_list)
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
