"""Tests for geometry optimisation calculation."""

import subprocess

from ase.build import bulk
import pytest

from aiida.common import datastructures
from aiida.engine import run
from aiida.orm import Bool, Str, StructureData
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData


def test_geomopt(fixture_sandbox, generate_calc_job, tmp_path, janus_code):
    """Test generating singlepoint calculation job"""
    # pylint:disable=line-too-long
    entry_point_name = "janus.opt"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.download(
            "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
            architecture="mace",
            cache_dir=tmp_path,
        ),
        "device": Str("cpu"),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    # pylint:disable=line-too-long
    cmdline_params = [
        "geomopt",
        "--arch",
        "mace",
        "--struct",
        "aiida.cif",
        "--device",
        "cpu",
        "--log",
        "aiida.log",
        "--calc-kwargs",
        f"{{'model': '{tmp_path}/mace/mace_mp_small.model', 'default_dtype': 'float64'}}",
        "--write-kwargs",
        "{'filename': 'aiida-results.xyz'}",
        "--traj",
        "aiida-traj.xyz",
        "--max-force",
        0.1,
    ]

    print(calc_info.codes_info[0].cmdline_params)
    print(cmdline_params)

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
        "aiida-traj.xyz",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == ["aiida.cif"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert len(calc_info.codes_info[0].cmdline_params) == len(cmdline_params)
    for x, y in zip((calc_info.codes_info[0].cmdline_params), (cmdline_params)):
        assert x == y
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_run_opt(tmp_path, janus_code):
    """Test running singlepoint calculation"""
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.download(
            "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
            architecture="mace",
            cache_dir=tmp_path,
        ),
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
