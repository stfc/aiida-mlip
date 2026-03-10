"""Example code for submitting training calculation."""

from __future__ import annotations

from pathlib import Path

from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile

# Add the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

ROOT_DIR = Path(__file__).resolve().parents[2]

# All the other parameters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance
mlip_config = JanusConfigfile(ROOT_DIR / "tests/calculations/configs/mlip_train.yml")

# Define calculation to run
TrainCalc = CalculationFactory("mlip.train")

# Run calculation
result, node = run_get_node(
    TrainCalc,
    code=code,
    metadata=metadata,
    mlip_config=mlip_config,
)
print(f"Printing results from calculation: {result}")
print(f"Printing node of calculation: {node}")
