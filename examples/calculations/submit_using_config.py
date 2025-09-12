"""Example code for submitting single point calculation."""

from __future__ import annotations

from pathlib import Path

from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import CalculationFactory

import aiida_mlip
from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_model, load_structure

# __file__ changes depending on where verdi run is called
DATA_PATH = Path(aiida_mlip.__file__).parent.parent / "tests" / "calculations"

# Add the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

# This structure will overwrite the one in the config file if present
structure = load_structure(DATA_PATH / "structures" / "NaCl.cif")

# This model will overwrite the one in the config file if present
model = load_model(
    model="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
    architecture="mace_mp",
)

# All the other paramenters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance
config = JanusConfigfile(DATA_PATH / "configs" / "config_janus.yml")

# Define calculation to run
SinglePointCalc = CalculationFactory("mlip.sp")

# Run calculation
result, node = run_get_node(
    SinglePointCalc,
    code=code,
    struct=structure,
    model=model,
    metadata=metadata,
    config=config,
)
print(f"Printing results from calculation: {result}")
print(f"Printing node of calculation: {node}")
