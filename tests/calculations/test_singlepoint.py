"""Tests for singlepoint calculation."""

import subprocess

from ase.build import bulk
import pytest

from aiida.common import datastructures
from aiida.engine import run
from aiida.orm import Str, StructureData
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData


def test_singlepoint(fixture_sandbox, generate_calc_job, tmp_path, janus_code):
    """Test generating singlepoint calculation job"""
    # pylint:disable=line-too-long
    entry_point_name = "janus.sp"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "calctype": Str("singlepoint"),
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
        "singlepoint",
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
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == ["aiida.cif"]
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_sp_error(fixture_sandbox, generate_calc_job, tmp_path, fixture_code):
    """Test singlepoint calculation with error input"""
    entry_point_name = "janus.sp"
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": fixture_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "calctype": Str("wrong_type"),
        "model": ModelData.download(
            "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
            architecture="mace",
            cache_dir=tmp_path,
        ),
        "device": Str("cpu"),
    }
    with pytest.raises(ValueError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_run_sp(tmp_path, janus_code):
    """Test running singlepoint calculation"""
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "calctype": Str("singlepoint"),
        "model": ModelData.download(
            "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
            architecture="mace",
            cache_dir=tmp_path,
        ),
        "device": Str("cpu"),
    }

    singlePointCalculation = CalculationFactory("janus.sp")
    result = run(singlePointCalculation, **inputs)

    assert "results_dict" in result
    obtained_res = result["results_dict"].get_dict()
    assert "log_output" in result
    assert "xyz_output" in result
    assert "std_output" in result
    assert obtained_res["info"]["energy"] == pytest.approx(-6.7575203839729)
    assert obtained_res["info"]["stress"][0][0] == pytest.approx(-0.005816546985101)


def test_example(example_file_path):
    """
    Test function to execute the example file with specific command arguments.
    """
    command = [
        "verdi",
        "run",
        example_file_path,
        "janus@localhost",
        "--calctype",
        "singlepoint",
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
