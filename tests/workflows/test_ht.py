"""Test for high-throughput WorkGraphs."""

from aiida.orm import SinglefileData, StructureData
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData
from aiida_mlip.workflows.ht_workgraph import get_ht_workgraph


def test_ht_singlepoint(janus_code, workflow_structure_folder, model_folder) -> None:
    """Test high throughput singlepoint calculation."""
    SinglepointCalc = CalculationFactory("mlip.sp")

    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "model": ModelData.from_local(model_file, architecture="mace"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
    }

    wg = get_ht_workgraph(
        calc=SinglepointCalc,
        folder=workflow_structure_folder,
        calc_inputs=inputs,
        final_struct_key="xyz_output",
    )

    wg.submit(wait=True)

    assert wg.state in ("CREATED", "WAITING", "FINISHED")

    for _ in range(30):
        if wg.state == "FINISHED":
            break
        wg.wait(10)
    assert wg.state == "FINISHED"
    assert wg.state == "FINISHED"

    assert isinstance(wg.process.outputs.final_structures.H2O, SinglefileData)
    assert isinstance(wg.process.outputs.final_structures.methane, SinglefileData)


def test_ht_geomopt(janus_code, workflow_structure_folder, model_folder) -> None:
    """Test high throughput geometry optimisation."""
    GeomoptCalc = CalculationFactory("mlip.opt")

    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "model": ModelData.from_local(model_file, architecture="mace"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
    }

    wg = get_ht_workgraph(
        calc=GeomoptCalc,
        folder=workflow_structure_folder,
        calc_inputs=inputs,
    )

    wg.submit(wait=True)

    assert wg.state in ("CREATED", "WAITING", "FINISHED")

    for _ in range(30):
        if wg.state == "FINISHED":
            break
        wg.wait(10)
    assert wg.state == "FINISHED"

    assert isinstance(wg.process.outputs.final_structures.H2O, StructureData)
    assert isinstance(wg.process.outputs.final_structures.methane, StructureData)