"""Tests for geometry optimisation calculation."""

import subprocess

from ase.build import bulk
from ase.io import read
import pytest

from aiida.common import datastructures
from aiida.engine import run
from aiida.orm import Dict, Str, StructureData
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData


def test_MD(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating singlepoint calculation job"""

    entry_point_name = "janus.md"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.local_file(model_file, architecture="mace"),
        "device": Str("cpu"),
        "ensemble": Str("nve"),
        "md_dict": Dict(
            {
                "temp": 300.0,
                "steps": 4,
                "traj-every": 1,
                "restart-every": 3,
                "stats-every": 1,
            }
        ),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        "md",
        "--arch",
        "mace",
        "--struct",
        "aiida.xyz",
        "--device",
        "cpu",
        "--log",
        "aiida.log",
        "--calc-kwargs",
        f"{{'model': '{model_file}', 'default_dtype': 'float64'}}",
        "--ensemble",
        "nve",
        "--temp",
        300.0,
        "--steps",
        4,
        "--traj-every",
        1,
        "--stats-every",
        1,
        "--restart-every",
        3,
        "--traj-file",
        "aiida-traj.xyz",
        "--stats-file",
        "aiida-stats.dat",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-stdout.txt",
        "aiida-traj.xyz",
        "aiida-stats.dat",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == ["aiida.xyz"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert len(calc_info.codes_info[0].cmdline_params) == len(cmdline_params)
    assert sorted(map(str, calc_info.codes_info[0].cmdline_params)) == sorted(
        map(str, cmdline_params)
    )
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_run_md(model_folder, structure_folder, janus_code):
    """Test running singlepoint calculation"""

    model_file = model_folder / "mace_mp_small.model"
    structure_file = structure_folder / "NaCl.cif"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=read(structure_file)),
        "model": ModelData.local_file(model_file, architecture="mace"),
        "device": Str("cpu"),
        "ensemble": Str("nve"),
        "md_dict": Dict(
            {
                "temp": 300.0,
                "steps": 3,
                "traj-every": 1,
                "restart-every": 3,
                "stats-every": 1,
            }
        ),
    }

    MDCalculation = CalculationFactory("janus.md")
    result = run(MDCalculation, **inputs)

    assert "final_structure" in result
    assert "traj_output" in result
    assert "traj_file" in result
    assert result["traj_output"].numsteps == 4
    assert result["traj_output"].get_step_data(1)[4][3][1] == pytest.approx(2.82)


def test_example_md(example_path):
    """
    Test function to execute the example file with specific command arguments.
    """
    example_file_path = example_path / "submit_md.py"
    command = ["verdi", "run", example_file_path, "janus@localhost"]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
