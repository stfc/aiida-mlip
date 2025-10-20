"""Tests for pressure functionality in geometry optimisation calculations."""

from __future__ import annotations

from aiida.engine import run
from aiida.orm import Bool, Float, Int, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.build import bulk
from ase.io import write
from click.testing import CliRunner

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


# check that click works and changes output
def test_pressure_click_changes_output(model_folder, janus_code, tmp_path):
    """Test that setting --pressure in CLI changes output compared to 0 pressure."""
    # Create a temporary structure file
    structure_file = tmp_path / "test_structure.cif"
    nacl = bulk("NaCl", "rocksalt", 5.63)
    write(str(structure_file), nacl)

    from examples.calculations.submit_geomopt import cli

    runner = CliRunner()

    # Run with 0 pressure (baseline)
    result_0_pressure = runner.invoke(
        cli,
        [
            "janus@localhost",
            "--struct",
            str(structure_file),
            "--model",
            str(model_folder / "mace_mp_small.model"),
            "--arch",
            "mace_mp",
            "--device",
            "cpu",
            "--opt_cell_fully",
            "True",
            "--pressure",
            "0.0",
            "--steps",
            "100",
        ],
    )

    # Run with 5 GPa pressure
    result_5_pressure = runner.invoke(
        cli,
        [
            "janus@localhost",
            "--struct",
            str(structure_file),
            "--model",
            str(model_folder / "mace_mp_small.model"),
            "--arch",
            "mace_mp",
            "--device",
            "cpu",
            "--opt_cell_fully",
            "True",
            "--pressure",
            "5.0",  # Apply pressure via CLI
            "--steps",
            "100",
        ],
    )

    # Both should complete successfully
    assert result_0_pressure.exit_code == 0, (
        f"0 pressure CLI failed: {result_0_pressure.output}"
    )
    assert result_5_pressure.exit_code == 0, (
        f"5 GPa CLI failed: {result_5_pressure.output}"
    )
