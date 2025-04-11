"""Example code for submitting single point calculation."""

from __future__ import annotations

from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure

# Add the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

# This structure will overwrite the one in the config file if present
structure = load_structure("../tests/calculations/structures/NaCl.cif")

# All the other paramenters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance
config = JanusConfigfile("../tests/calculations/configs/config_janus.yaml")

# Define calculation to run
SinglepointCalc = CalculationFactory("mlip.sp")

# Run calculation
result, node = run_get_node(
    SinglepointCalc,
    code=code,
    struct=structure,
    metadata=metadata,
    config=config,
)
print(f"Printing results from calculation: {result}")
print(f"Printing node of calculation: {node}")
