"""Tests for phonon calculations calculation."""

from __future__ import annotations

from aiida.common import InputValidationError, datastructures
from aiida.engine import run
from aiida.orm import Bool, Float, Int, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
import numpy as np
import pytest

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData


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
        "displacement": Float(0.01),
        "nqpoints": Int(51),
        "dos": Bool(False),
        "pdos": Bool(False),
        "bands": Bool(False),
        "no_hdf5": Bool(False),
        "symmetrize": Bool(False),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        "phonons",
        "--struct",
        "aiida.xyz",
        "--log",
        "aiida.log",
        "--summary",
        "phonon-summary.yml",
        "--device",
        "cpu",
        "--arch",
        "mace",
        "--model",
        "mlff.model",
        "--file-prefix",
        "aiida",
        "--supercell",
        "2 2 2",
        "--displacement",
        "0.01",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida-stdout.txt",
        "aiida.log",
        "phonon-summary.yml",
        "aiida-phonopy.yml",
        "aiida-force_constants.hdf5",
        "aiida-dos.dat",
        "aiida-pdos.dat",
        "aiida-auto_bands.yml.xz",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ["aiida.xyz", "mlff.model"]
    )

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    calc_str = map(str, calc_info.codes_info[0].cmdline_params)
    assert sorted(calc_str) == sorted(cmdline_params)
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
        "displacement": Float(0.01),
        "nqpoints": Int(51),
        "dos": Bool(False),
        "pdos": Bool(False),
        "bands": Bool(False),
        "no_hdf5": Bool(True),
        "symmetrize": Bool(False),
    }

    PhononCalc = CalculationFactory("mlip.ph")
    result = run(PhononCalc, **inputs)

    assert "results_dict" in result
    assert "phonon_output" in result

    lattice_vectors = result["results_dict"].get_dict()["primitive_cell"]["lattice"]
    supercell_matrix = result["results_dict"].get_dict()["supercell_matrix"]
    fc = result["results_dict"].get_dict()["force_constants"]["elements"]

    assert lattice_vectors[0][1] == pytest.approx(2.815, rel=1.0e-4, abs=1.0e-4)
    assert (np.diag(lattice_vectors) == 0.0).all()
    assert (np.diag(supercell_matrix) == 2).all()
    assert fc[0][0][0] == pytest.approx(2.1912727829831, rel=1.0e-4, abs=1.0e-4)


def test_run_supercell(model_folder, janus_code):
    """Test running singlepoint calculation."""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "supercell": Str("3 3 3"),
        "displacement": Float(0.01),
        "nqpoints": Int(51),
        "dos": Bool(False),
        "pdos": Bool(False),
        "bands": Bool(False),
        "no_hdf5": Bool(True),
        "symmetrize": Bool(False),
    }

    PhononCalc = CalculationFactory("mlip.ph")
    result = run(PhononCalc, **inputs)

    assert "results_dict" in result
    assert "phonon_output" in result

    lattice_vectors = result["results_dict"].get_dict()["primitive_cell"]["lattice"]
    supercell_matrix = result["results_dict"].get_dict()["supercell_matrix"]
    fc = result["results_dict"].get_dict()["force_constants"]["elements"]

    assert lattice_vectors[0][1] == pytest.approx(2.815, rel=1.0e-4, abs=1.0e-4)
    assert (np.diag(lattice_vectors) == 0.0).all()
    assert (np.diag(supercell_matrix) == 3).all()
    assert fc[0][0][0] == pytest.approx(2.2240705622516, rel=1.0e-4, abs=1.0e-4)



def test_output_files(fixture_sandbox, generate_calc_job, janus_code, model_folder):
    """Est setting log and summary output files."""
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
        "displacement": Float(0.01),
        "dos": Bool(True),
        "pdos": Bool(True),
        "bands": Bool(True),
        "no_hdf5": Bool(False),
        "symmetrize": Bool(False),
        "nqpoints": Int(51),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        "phonons",
        "--struct",
        "aiida.xyz",
        "--log",
        "aiida.log",
        "--summary",
        "phonon-summary.yml",
        "--device",
        "cpu",
        "--arch",
        "mace",
        "--model",
        "mlff.model",
        "--file-prefix",
        "aiida",
        "--supercell",
        "2 2 2",
        "--displacement",
        "0.01",
        "--bands",
        "--dos",
        "--pdos",
        "--n-qpoints",
        "51",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida-stdout.txt",
        "aiida.log",
        "phonon-summary.yml",
        "aiida-phonopy.yml",
        "aiida-force_constants.hdf5",
        "aiida-dos.dat",
        "aiida-pdos.dat",
        "aiida-auto_bands.yml.xz",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ["aiida.xyz", "mlff.model"]
    )
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    calc_str = map(str, calc_info.codes_info[0].cmdline_params)
    assert sorted(calc_str) == sorted(cmdline_params)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
