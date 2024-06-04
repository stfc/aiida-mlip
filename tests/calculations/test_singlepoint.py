"""Tests for singlepoint calculation."""

from pathlib import Path
import subprocess

from ase.build import bulk
from ase.io import write
import pytest

from aiida.common import InputValidationError, datastructures
from aiida.engine import run
from aiida.orm import Str, StructureData
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData


def test_singlepoint(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating singlepoint calculation job"""

    entry_point_name = "mlip.sp"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.local_file(model_file, architecture="mace"),
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
        f"{{'default_dtype': 'float64', 'model': '{model_file}'}}",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert fixture_sandbox.get_content_list() == ["aiida.xyz"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_singlepoint_model_download(fixture_sandbox, generate_calc_job, janus_code):
    """Test generating singlepoint calculation job."""

    entry_point_name = "mlip.sp"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "device": Str("cpu"),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert fixture_sandbox.get_content_list() == ["aiida.xyz"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_sp_nostruct(fixture_sandbox, generate_calc_job, model_folder, janus_code):
    """Test singlepoint calculation with error input"""
    entry_point_name = "mlip.sp"
    model_file = model_folder / "mace_mp_small.model"
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "model": ModelData.local_file(model_file, architecture="mace"),
        "device": Str("cpu"),
    }
    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_sp_nomodel(fixture_sandbox, generate_calc_job, config_folder, janus_code):
    """Test singlepoint calculation with error input"""
    entry_point_name = "mlip.sp"

    nacl = bulk("NaCl", "rocksalt", a=5.63)
    write("NaCl.cif", nacl)

    inputs = {
        "code": janus_code,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "config": JanusConfigfile(config_folder / "config_error.yml"),
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    Path("NaCl.cif").unlink()


def test_run_sp(model_folder, janus_code):
    """Test running singlepoint calculation"""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.local_file(model_file, architecture="mace"),
        "device": Str("cpu"),
    }

    singlePointCalculation = CalculationFactory("mlip.sp")
    result = run(singlePointCalculation, **inputs)
    assert "results_dict" in result
    obtained_res = result["results_dict"].get_dict()
    assert "xyz_output" in result
    assert obtained_res["info"]["mace_energy"] == pytest.approx(-6.7575203839729)
    assert obtained_res["info"]["mace_stress"][0] == pytest.approx(-0.005816546985101)


def test_example(example_path):
    """
    Test function to run md calculation through the use of the example file provided.
    """

    example_file_path = example_path / "submit_singlepoint.py"
    command = ["verdi", "run", example_file_path, "janus@localhost"]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
