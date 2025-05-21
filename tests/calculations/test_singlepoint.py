"""Tests for singlepoint calculation."""

from __future__ import annotations

import subprocess

from aiida.common import InputValidationError, datastructures
from aiida.engine import run
from aiida.orm import Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
import pytest

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData


def test_singlepoint(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating singlepoint calculation job."""
    entry_point_name = "mlip.sp"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        "singlepoint",
        "--arch",
        "mace",
        "--struct",
        "aiida.xyz",
        "--device",
        "cpu",
        "--log",
        "aiida.log",
        "--out",
        "aiida-results.xyz",
        "--calc-kwargs",
        "{'default_dtype': 'float64', 'model': 'mlff.model'}",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ["aiida.xyz", "mlff.model"]
    )
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_sp_nostruct(fixture_sandbox, generate_calc_job, model_folder, janus_code):
    """Test singlepoint calculation with error input."""
    entry_point_name = "mlip.sp"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
    }
    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_sp_nomodel(fixture_sandbox, generate_calc_job, config_folder, janus_code):
    """Test singlepoint calculation with missing model."""
    entry_point_name = "mlip.sp"

    inputs = {
        "code": janus_code,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "config": JanusConfigfile(config_folder / "config_nomodel.yml"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_sp_noarch(fixture_sandbox, generate_calc_job, config_folder, janus_code):
    """Test singlepoint calculation with missing architecture."""
    entry_point_name = "mlip.sp"

    inputs = {
        "code": janus_code,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "config": JanusConfigfile(config_folder / "config_noarch.yml"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_two_arch(fixture_sandbox, generate_calc_job, model_folder, janus_code):
    """Test singlepoint calculation with two defined architectures."""
    entry_point_name = "mlip.sp"
    model_file = model_folder / "mace_mp_small.model"

    inputs = {
        "code": janus_code,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "model": ModelData.from_local(model_file, architecture="mace_mp"),
        "arch": Str("chgnet"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_run_sp(model_folder, janus_code):
    """Test running singlepoint calculation."""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
    }

    SinglepointCalc = CalculationFactory("mlip.sp")
    result = run(SinglepointCalc, **inputs)
    assert "results_dict" in result
    obtained_res = result["results_dict"].get_dict()
    assert "xyz_output" in result
    assert obtained_res["info"]["mace_energy"] == pytest.approx(-6.7575203839729)
    assert obtained_res["info"]["mace_stress"][0] == pytest.approx(-0.005816546985101)


def test_example(example_path, janus_code, aiida_profile):
    """Test function to run singlepoint calculation using the example file provided."""
    example_file_path = example_path / "submit_singlepoint.py"
    profile = aiida_profile
    command = [
        "verdi",
        "profile",
        "set-default",
        profile.name,
    ]
    subprocess.run(command, capture_output=True, text=True, check=False)
    command = [
        "verdi",
        "run",
        example_file_path,
        f"{janus_code.label}@{janus_code.computer.label}",
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
    assert "results from calculation:" in result.stdout
    assert "'results_dict': <Dict: uuid:" in result.stdout
    assert "'xyz_output': <SinglefileData: uuid:" in result.stdout
