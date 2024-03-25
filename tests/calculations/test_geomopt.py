"""Tests for geometry optimisation calculation."""

import subprocess

from ase.build import bulk
import pytest

from aiida.common import datastructures
from aiida.engine import run
from aiida.orm import Bool, Str, StructureData
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData


def test_geomopt(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating singlepoint calculation job"""

    entry_point_name = "janus.opt"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.local_file(model_file, architecture="mace"),
        "device": Str("cpu"),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        "geomopt",
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
        f"{{'model': '{model_file}', 'default_dtype': 'float64'}}",
        "--traj",
        "aiida-traj.xyz",
        "--max-force",
        0.1,
        "--steps",
        1000,
        "--opt-kwargs",
        {},
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
        "aiida-traj.xyz",
    ]

    print(calc_info.codes_info[0].cmdline_params)
    print(cmdline_params)

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == ["aiida.xyz"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert len(calc_info.codes_info[0].cmdline_params) == len(cmdline_params)
    for x, y in zip((calc_info.codes_info[0].cmdline_params), (cmdline_params)):
        assert x == y
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_run_opt(model_folder, janus_code):
    """Test running singlepoint calculation"""

    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.local_file(model_file, architecture="mace"),
        "device": Str("cpu"),
        "fully_opt": Bool(True),
    }

    geomoptCalculation = CalculationFactory("janus.opt")
    result = run(geomoptCalculation, **inputs)

    assert "results_dict" in result
    assert "final_structure" in result
    assert "traj_output" in result
    assert "traj_file" in result
    assert result["traj_output"].numsteps == 3
    assert result["final_structure"].cell[0][0] == pytest.approx(4.0223130461422)


def test_example_opt(example_path):
    """
    Test function to execute the example file with specific command arguments.
    """
    example_file_path = example_path / "submit_geomopt.py"
    command = ["verdi", "run", example_file_path, "janus@localhost"]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
