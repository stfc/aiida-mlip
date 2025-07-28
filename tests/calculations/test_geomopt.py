"""Tests for geometry optimisation calculation."""

from __future__ import annotations

import subprocess

from aiida.common import datastructures
from aiida.engine import run
from aiida.orm import Bool, Float, Int, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
import pytest

from aiida_mlip.data.model import ModelData


def test_geomopt(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating geomopt calculation job."""
    entry_point_name = "mlip.opt"
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
        "{'default_dtype': 'float64', 'model': 'mlff.model'}",
        "--minimize-kwargs",
        "{'traj_kwargs': {'filename': 'aiida-traj.xyz'}}",
        "--write-traj",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
        "aiida-traj.xyz",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ["aiida.xyz", "mlff.model"]
    )
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert len(calc_info.codes_info[0].cmdline_params) == len(cmdline_params)
    assert sorted(map(str, calc_info.codes_info[0].cmdline_params)) == sorted(
        map(str, cmdline_params)
    )
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_run_opt(model_folder, janus_code):
    """Test running geomopt calculation."""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "opt_cell_fully": Bool(True),
        "fmax": Float(0.1),
        "steps": Int(1000),
    }

    GeomoptCalc = CalculationFactory("mlip.opt")
    result = run(GeomoptCalc, **inputs)
    assert "results_dict" in result
    assert "final_structure" in result
    assert "traj_output" in result
    assert "traj_file" in result
    assert result["traj_output"].numsteps == 3
    assert result["final_structure"].cell[0][1] == pytest.approx(2.8438848145858)
    assert result["xyz_output"].filename == "aiida-results.xyz"


def test_cli_kwargs(model_folder, janus_code):
    """Test that the command line arguments are correctly set."""
    model_file = model_folder / "mace_mp_small.model"
    minimize_kwargs = {
        "traj_kwargs": {"filename": "aiida-traj.xyz"},
        "filter_kwargs": {"constant_volume": True},
    }
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "opt_cell_fully": Bool(True),
        "fmax": Float(0.1),
        "steps": Int(1000),
        "minimize_kwargs": minimize_kwargs,
    }

    GeomoptCalc = CalculationFactory("mlip.opt")
    result = run(GeomoptCalc, **inputs)
    assert result["final_structure"].cell[0][1] == pytest.approx(2.815)
    assert result["traj_file"].filename == "aiida-traj.xyz"


def test_example_opt(example_path, janus_code):
    """Test function to run geometry optimization using the example file provided."""
    example_file_path = example_path / "submit_geomopt.py"
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
    assert "'traj_file': <SinglefileData: uuid:" in result.stdout
    assert "'final_structure': <StructureData: uuid:" in result.stdout
