"""Tests for descriptors calculation."""

from __future__ import annotations

import subprocess

from aiida.common import datastructures
from aiida.engine import run
from aiida.orm import Bool, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
import pytest

from aiida_mlip.data.model import ModelData


def test_descriptors(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating descriptors calculation job."""
    entry_point_name = "mlip.descriptors"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "invariants_only": Bool(True),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        "descriptors",
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
        "--invariants-only",
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
    assert len(calc_info.codes_info[0].cmdline_params) == len(cmdline_params)
    assert sorted(map(str, calc_info.codes_info[0].cmdline_params)) == sorted(
        map(str, cmdline_params)
    )
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)


def test_run_descriptors(model_folder, janus_code):
    """Test running descriptors calculation."""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "precision": Str("float64"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "invariants_only": Bool(False),
        "calc_per_element": Bool(True),
        "calc_per_atom": Bool(True),
    }

    DescriptorsCalc = CalculationFactory("mlip.descriptors")
    result = run(DescriptorsCalc, **inputs)

    assert "xyz_output" in result
    assert result["xyz_output"].filename == "aiida-results.xyz"

    assert "results_dict" in result
    obtained_res = result["results_dict"].get_dict()
    assert obtained_res["info"]["mace_descriptor"] == pytest.approx(-0.0056343183)
    assert obtained_res["info"]["mace_Cl_descriptor"] == pytest.approx(-0.0091900828)
    assert obtained_res["info"]["mace_Na_descriptor"] == pytest.approx(-0.0020785538)
    assert obtained_res["mace_descriptors"] == pytest.approx([-0.00207855, -0.00919008])


def test_example_descriptors(example_path, janus_code):
    """Test running descriptors calculation using the example file provided."""
    example_file_path = example_path / "submit_descriptors.py"
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
