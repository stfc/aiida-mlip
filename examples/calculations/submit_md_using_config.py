"""Example code for submitting single point calculation."""

from __future__ import annotations

from pathlib import Path

from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure

# And the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

# This structure will overwrite the one in the config file if present
structure = load_structure()

# All the other paramenters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance


config = JanusConfigfile(
    Path("../../../tests/calculations/configs/config_janus_md.yml")
    .expanduser()
    .resolve()
)

# Define calculation to run
MDCalculation = CalculationFactory("mlip.md")

# Run calculation
result, node = run_get_node(
    MDCalculation, code=code, struct=structure, metadata=metadata, config=config
)
print(f"Printing results from calculation: {result}")
print(f"Printing node of calculation: {node}")
