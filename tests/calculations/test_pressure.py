"""Tests for pressure functionality in geometry optimisation calculations."""

from __future__ import annotations

import subprocess

from aiida.engine import run
from aiida.orm import Bool, Float, Int, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
from ase.io import write

from aiida_mlip.data.model import ModelData


# Structure changes depending on pressure applied
def test_pressure_optimization(model_folder, janus_code):
    """Test geometry optimization with external pressure applied."""
    model_file = model_folder / "mace_mp_small.model"

    # Test with different pressure values
    pressure_values = [0.0, 5.0, 10.0]
    results = {}

    for pressure in pressure_values:
        inputs = {
            "metadata": {"options": {"resources": {"num_machines": 1}}},
            "code": janus_code,
            "arch": Str("mace"),
            "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
            "model": ModelData.from_local(model_file, architecture="mace"),
            "device": Str("cpu"),
            "opt_cell_fully": Bool(True),
            "fmax": Float(0.1),
            "steps": Int(1000),
            "pressure": Float(pressure),
        }

        GeomoptCalc = CalculationFactory("mlip.opt")
        result = run(GeomoptCalc, **inputs)
        results[pressure] = result

        # Basic assertions
        assert "results_dict" in result
        assert "final_structure" in result
        assert "traj_output" in result

    # Test that pressure affects the final structure
    # Higher pressure should generally lead to smaller cell volumes
    vol_0gpa = results[0.0]["final_structure"].get_cell_volume()
    vol_5gpa = results[5.0]["final_structure"].get_cell_volume()
    vol_10gpa = results[10.0]["final_structure"].get_cell_volume()

    # Assert that volume decreases with increasing pressure
    assert vol_5gpa < vol_0gpa, "5 GPa should compress the structure compared to 0 GPa"
    assert vol_10gpa < vol_5gpa, "10 GPa should compress the structure more than 5 GPa"


# Test command line generation
def test_pressure_command_line_generation(
    fixture_sandbox, generate_calc_job, janus_code, model_folder
):
    """Test that pressure parameter is correctly passed to command line."""
    entry_point_name = "mlip.opt"
    model_file = model_folder / "mace_mp_small.model"
    test_pressure = 7.5

    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "arch": Str("mace"),
        "struct": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "model": ModelData.from_local(model_file, architecture="mace"),
        "device": Str("cpu"),
        "pressure": Float(test_pressure),
        "opt_cell_fully": Bool(True),  # Cell optimization must be enabled for pressure
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    # Check that pressure appears in command line parameters
    cmdline_params = calc_info.codes_info[0].cmdline_params
    cmdline_str = " ".join(map(str, cmdline_params))

    assert "--pressure" in cmdline_str
    assert str(test_pressure) in cmdline_str


def test_pressure_verdi_run(example_path, janus_code, tmp_path):
    """Test pressure option click."""
    # Create a temporary structure file
    structure_file = tmp_path / "test_structure.cif"
    nacl = bulk("NaCl", "rocksalt", 5.63)
    write(str(structure_file), nacl)

    example_file_path = example_path / "submit_geomopt.py"

    # Run with 0 pressure (baseline)
    command_0_pressure = [
        "verdi",
        "run",
        example_file_path,
        f"{janus_code.label}@{janus_code.computer.label}",
        "--struct",
        str(structure_file),
        "--pressure",
        "0.0",
        "--opt_cell_fully",
        "True",
        "--steps",
        "100",
    ]

    # Run with 5 GPa pressure
    command_5_pressure = [
        "verdi",
        "run",
        example_file_path,
        f"{janus_code.label}@{janus_code.computer.label}",
        "--struct",
        str(structure_file),
        "--pressure",
        "5.0",
        "--opt_cell_fully",
        "True",
        "--steps",
        "100",
    ]

    # Execute both commands
    result_0_pressure = subprocess.run(
        command_0_pressure, capture_output=True, text=True, check=False
    )
    result_5_pressure = subprocess.run(
        command_5_pressure, capture_output=True, text=True, check=False
    )

    # Both should complete successfully (like test_example_opt)
    assert result_0_pressure.returncode == 0
    assert result_5_pressure.returncode == 0
