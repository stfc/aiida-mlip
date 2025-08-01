"""Test for high-throughput WorkGraphs."""

from __future__ import annotations

from aiida.orm import SinglefileData, StructureData
from aiida.plugins import CalculationFactory
import pytest

from aiida_mlip.data.model import ModelData
from aiida_mlip.workflows.ht_workgraph import build_ht_calc


def test_ht_singlepoint(janus_code, workflow_structure_folder, model_folder) -> None:
    """Test high throughput singlepoint calculation."""
    SinglepointCalc = CalculationFactory("mlip.sp")

    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "model": ModelData.from_local(model_file, architecture="mace"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
    }

    wg = build_ht_calc(
        calc=SinglepointCalc,
        folder=workflow_structure_folder,
        calc_inputs=inputs,
        final_struct_key="xyz_output",
    )

    wg.run()

    assert wg.state == "FINISHED"

    assert isinstance(wg.outputs.final_structure.H2O.value, SinglefileData)
    assert isinstance(wg.outputs.final_structure.methane.value, SinglefileData)


def test_ht_invalid_path(janus_code, workflow_invalid_folder, model_folder) -> None:
    """Test invalid path for high throughput calculation."""
    SinglepointCalc = CalculationFactory("mlip.sp")

    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "model": ModelData.from_local(model_file, architecture="mace"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
    }

    with pytest.raises(FileNotFoundError):
        build_ht_calc(
            calc=SinglepointCalc,
            folder=workflow_invalid_folder,
            calc_inputs=inputs,
            final_struct_key="xyz_output",
        )


def test_ht_geomopt(janus_code, workflow_structure_folder, model_folder) -> None:
    """Test high throughput geometry optimisation."""
    GeomoptCalc = CalculationFactory("mlip.opt")

    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "model": ModelData.from_local(model_file, architecture="mace"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
    }

    wg = build_ht_calc(
        calc=GeomoptCalc,
        folder=workflow_structure_folder,
        calc_inputs=inputs,
    )

    wg.run()

    assert wg.state == "FINISHED"

    assert isinstance(wg.process.outputs.final_structure.H2O, StructureData)
    assert isinstance(wg.process.outputs.final_structure.methane, StructureData)
