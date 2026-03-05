"""Tests for phonon calculations calculation."""

from __future__ import annotations

import subprocess

from aiida.common import InputValidationError, datastructures
from aiida.engine import run
from aiida.orm import Dict, Str, Bool, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
from ase.io import write
import pytest
import yaml

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData
from tests.utils import chdir


def test_phonon(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test generating phonon calculation job."""
    entry_point_name = "mlip.ph"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "supercell": Str("2 2 2"),
        "minimize" : Bool(False)
    }


    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        'phonons', '--struct', 'aiida.xyz', '--log', 'aiida.log', '--summary', 'phonon-summary.yml', '--device', 'cpu', 
        '--arch', 'mace', '--model', 'mlff.model', '--no-hdf5', '--file-prefix', 'aiida', '--supercell', '2 2 2']
    
    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-stdout.txt",
        "phonon-summary.yml", 
        "aiida-phonopy.yml"
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ["aiida.xyz", "mlff.model"]
    )
    
    assert isinstance(calc_info, datastructures.CalcInfo)
    
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    

def test_ph_nostruct(fixture_sandbox, generate_calc_job, model_folder, janus_code):
    """Test singlepoint calculation with error input."""
    entry_point_name = "mlip.ph"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
    }
    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_ph_nomodel(fixture_sandbox, generate_calc_job, config_folder, janus_code):
    """Test singlepoint calculation with missing model."""
    entry_point_name = "mlip.ph"

    inputs = {
        "code": janus_code,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "config": JanusConfigfile(config_folder / "config_nomodel.yml"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
    }

    with pytest.raises(InputValidationError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_ph_noarch(fixture_sandbox, generate_calc_job, config_folder, janus_code):
    """Test singlepoint calculation with missing architecture."""
    entry_point_name = "mlip.ph"

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
    entry_point_name = "mlip.ph"
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


def test_run_ph(model_folder, janus_code):
    """Test running singlepoint calculation."""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "supercell": Str("2 2 2"),
        "minimize" : Bool(False),
    }

    PhononCalc = CalculationFactory("mlip.ph")
    result = run(PhononCalc, **inputs)
    assert "results_dict" in result
    obtained_res = result["results_dict"].get_dict()
    assert "xyz_output" in result
    
def test_example(example_path, janus_code):
    """Test function to run phonon calculation using the default keywords."""
    example_file_path = example_path / "submit_phonon.py"
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
    assert "super cell" in result.stdout

    result = {}
    with open('data.yml', 'r') as file:
        result = yaml.safe_load(file)

    lattice_vectors = result["primitive_cell"]["lattice"]
    
    supercell_matrix = result['supercell_matrix']
    
    fc = result['force_constants']['elements']
    
    assert lattice_vectors[0][1] == pytest.approx(2.815, rel=1.0e-4, abs=1.0e-4)
    assert supercell_matrix[0][0] == pytest.approx(2)
    assert fc[0][0][0] == pytest.approx(2.1912727829831, rel=1.0e-4, abs=1.0e-4)

def test_example_supercell(example_path, janus_code):
    """Test function to run phonon calculation using the supercell command."""
    example_file_path = example_path / "submit_phonon.py"
    command = [
        "verdi",
        "run",
        example_file_path,
        f"{janus_code.label}@{janus_code.computer.label}",
        "--supercell", "3 3 3",
    ]

    # Execute the command

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
    assert "super cell" in result.stdout

    result = {}
    with open('data.yml', 'r') as file:
        result = yaml.safe_load(file)

    lattice_vectors = result["primitive_cell"]["lattice"]

    supercell_matrix = result['supercell_matrix']

    fc = result['force_constants']['elements']

    assert lattice_vectors[0][1] == pytest.approx(2.815, rel=1.0e-4, abs=1.0e-4)
    assert supercell_matrix[0][0] == pytest.approx(3)
    assert fc[0][0][0] == pytest.approx(2.2240705622516, rel=1.0e-4, abs=1.0e-4)

def test_example(example_path, janus_code):
    """Test function to run phonon calculation post minimisation of the NaCl cell."""
    example_file_path = example_path / "submit_phonon.py"
    command = [
        "verdi",
        "run",
        example_file_path,
        f"{janus_code.label}@{janus_code.computer.label}",
        "--minimize",

    ]

    # Execute the command

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stderr == ""
    assert result.returncode == 0
    assert "super cell" in result.stdout

    result = {}
    with open('data.yml', 'r') as file:
        result = yaml.safe_load(file)

    #get some variables
    lattice_vectors = result["primitive_cell"]["lattice"]
    
    supercell_matrix = result['supercell_matrix']
    
    fc = result['force_constants']['elements']
    
    #check that they are what is expected
    assert lattice_vectors[0][1] == pytest.approx(2.843884, rel=1.0e-4, abs=1.0e-4)
    assert supercell_matrix[0][0] == pytest.approx(2)
    assert fc[0][0][0] == pytest.approx(1.9563971755818, rel=1.0e-4, abs=1.0e-4)

def test_output_files(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Test setting log and summary output files."""
    entry_point_name = "mlip.ph"
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "supercell": Str("2 2 2"),
        "minimize" : Bool(False)
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        'phonons', '--struct', 'aiida.xyz', '--log', 'aiida.log', '--summary', 'phonon-summary.yml', '--device', 'cpu', 
        '--arch', 'mace', '--model', 'mlff.model', '--no-hdf5', '--file-prefix', 'aiida', '--supercell', '2 2 2']

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-stdout.txt",
        "phonon-summary.yml", 
        "aiida-phonopy.yml",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ["aiida.xyz", "mlff.model"]
    )
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
