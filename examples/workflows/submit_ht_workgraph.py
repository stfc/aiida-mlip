"""Example submission for high throughput workgraph."""

from __future__ import annotations

from pathlib import Path

from aiida.orm import load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData
from aiida_mlip.workflows.ht_workgraph import get_ht_workgraph

SinglepointCalc = CalculationFactory("mlip.sp")

inputs = {
    "model": ModelData.from_local(
        "./tests/calculations/configs/test.model",
        architecture="mace_mp",
    ),
    "metadata": {"options": {"resources": {"num_machines": 1}}},
    "code": load_code("janus@localhost"),
}

wg = get_ht_workgraph(
    calc=SinglepointCalc,
    folder=Path("./tests/workflows/structures/"),
    calc_inputs=inputs,
    final_struct_key="xyz_output",
    max_number_jobs=10,
)

wg.submit()
