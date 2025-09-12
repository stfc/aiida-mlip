"""Tests for geometry optimisation calculation."""

from __future__ import annotations

from pathlib import Path
import subprocess

from aiida.common import datastructures
from aiida.engine import run_get_node
from aiida.orm import Dict, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
from ase.io import read, write
import pytest

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData
from tests.utils import chdir


def test_md(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating MD calculation job."""
    entry_point_name = "mlip.md"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "calc_kwargs": Dict({"default_dtype": "float64"}),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "ensemble": Str("nve"),
        "md_kwargs": Dict(
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
        "--model",
        "mlff.model",
        "--struct",
        "aiida.xyz",
        "--device",
        "cpu",
        "--log",
        "aiida.log",
        "--summary",
        "md-summary.yml",
        "--calc-kwargs",
        "{'default_dtype': 'float64'}",
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
        "md-summary.yml",
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


def test_md_with_config(
    fixture_sandbox,
    generate_calc_job,
    janus_code,
    model_folder,
    config_folder,
    tmp_path,
):
    """Test generating MD calculation job."""
    with chdir(tmp_path):
        # Create a temporary cif file to use as input
        nacl = bulk("NaCl", "rocksalt", a=5.63)
        write("NaCl.cif", nacl)

        entry_point_name = "mlip.md"
        model_file = model_folder / "mace_mp_small.model"
        inputs = {
            "code": janus_code,
            "model": ModelData.from_local(file=model_file, architecture="mace"),
            "metadata": {"options": {"resources": {"num_machines": 1}}},
            "config": JanusConfigfile(config_folder / "config_janus_md.yml"),
            "calc_kwargs": Dict({"default_dtype": "float64"}),
        }

        calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

        cmdline_params = [
            "md",
            "--struct",
            "aiida.xyz",
            "--log",
            "aiida.log",
            "--arch",
            "mace",
            "--model",
            "mlff.model",
            "--calc-kwargs",
            "{'default_dtype': 'float64'}",
            "--config",
            "config.yaml",
            "--ensemble",
            "nvt",
            "--summary",
            "md-summary.yml",
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
            "md-summary.yml",
        ]

        # Check the attributes of the returned `CalcInfo`
        assert sorted(fixture_sandbox.get_content_list()) == sorted(
            [
                "aiida.xyz",
                "config.yaml",
                "mlff.model",
            ]
        )
        assert isinstance(calc_info, datastructures.CalcInfo)
        assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
        assert len(calc_info.codes_info[0].cmdline_params) == len(cmdline_params)
        assert sorted(map(str, calc_info.codes_info[0].cmdline_params)) == sorted(
            map(str, cmdline_params)
        )
        assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)

        Path("NaCl.cif").unlink(missing_ok=True)


def test_run_md(model_folder, structure_folder, janus_code):
    """Test running molecular dynamics calculation."""
    model_file = model_folder / "mace_mp_small.model"
    structure_file = structure_folder / "NaCl.cif"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "calc_kwargs": Dict({"default_dtype": "float64"}),
        "struct": StructureData(ase=read(structure_file)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "ensemble": Str("nve"),
        "md_kwargs": Dict(
            {
                "temp": 300.0,
                "steps": 3,
                "traj-every": 1,
                "restart-every": 3,
                "stats-every": 1,
            }
        ),
    }

    MDCalc = CalculationFactory("mlip.md")
    result, node = run_get_node(MDCalc, **inputs)

    assert "final_structure" in result
    assert "traj_output" in result
    assert "traj_file" in result
    assert "results_dict" in result
    assert result["traj_output"].numsteps == 4
    assert node.outputs.final_structure.get_cell_volume() == pytest.approx(179.406)


def test_run_md_config(model_folder, janus_code, config_folder, tmp_path):
    """Test running molecular dynamics simulation with config file."""
    with chdir(tmp_path):
        # Create a temporary cif file to use as input
        nacl = bulk("NaCl", "rocksalt", a=5.63)
        write("NaCl.cif", nacl)

        model_file = model_folder / "mace_mp_small.model"
        inputs = {
            "metadata": {"options": {"resources": {"num_machines": 1}}},
            "code": janus_code,
            "model": ModelData.from_local(model_file, architecture="mace"),
            "config": JanusConfigfile(config_folder / "config_janus_md.yml"),
        }

        MDCalc = CalculationFactory("mlip.md")
        result, node = run_get_node(MDCalc, **inputs)

        assert "final_structure" in result
        assert "traj_output" in result
        assert "traj_file" in result
        assert "results_dict" in result
        assert result["traj_output"].numsteps == 4
        assert node.outputs.final_structure.get_cell_volume() == pytest.approx(
            44.61338675
        )

        Path("NaCl.cif").unlink(missing_ok=True)


def test_example_md(example_path, janus_code):
    """Test function to run MD calculation using the example file provided."""
    example_file_path = example_path / "submit_md.py"
    command = [
        "verdi",
        "run",
        example_file_path,
        f"{janus_code.label}@{janus_code.computer.label}",
        "--md_dict_str",
        "{'steps': 10, 'traj-every': 1}",
    ]
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
    assert "results from calculation:" in result.stdout
    assert "'results_dict': <Dict: uuid:" in result.stdout
    assert "'traj_output': <TrajectoryData: uuid:" in result.stdout
    assert "'final_structure': <StructureData: uuid" in result.stdout
    assert "'stats_file': <SinglefileData: uuid" in result.stdout
