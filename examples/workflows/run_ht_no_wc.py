"""Example code for submitting high-throughpout calculation without a Workchain."""

import csv
from pathlib import Path
import sys
import time

from aiida.common import NotExistent
from aiida.engine import run_get_pk, submit
from aiida.orm import load_code, load_group, load_node
from aiida.plugins import CalculationFactory
import click

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData
from aiida_mlip.helpers.help_load import load_structure


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def run_ht(folder, config, calc, output_filename, code, group, launch):
    """Run high throughput calculation using the parameters from the cli."""
    # Add the required inputs for aiida
    metadata = {"options": {"resources": {"num_machines": 1}}}

    # All the other paramenters we want them from the config file
    # We want to pass it as a AiiDA data type for the provenance
    conf = JanusConfigfile(config)
    # Define calculation to run
    Calculation = CalculationFactory(f"mlip.{calc}")
    # pylint: disable=line-too-long
    model = ModelData.from_uri(
        uri="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        cache_dir="models",
        architecture="mace_mp",
        filename="small.model",
    )
    list_of_nodes = []
    p = Path(folder)
    for child in p.glob("**/*"):
        if child.name.endswith("cif"):
            print(child.name)
            metadata["label"] = f"{child.name}"
            # This structure will overwrite the one in the config file if present
            structure = load_structure(child.absolute())
            # Run calculation
            if launch == "run_get_pk":
                result, pk = run_get_pk(
                    Calculation,
                    code=code,
                    struct=structure,
                    metadata=metadata,
                    config=conf,
                    model=model,
                )
                list_of_nodes.append(pk)

                group.add_nodes(load_node(pk))
                time.sleep(1)
                print(f"Printing results from calculation: {result}")

            if launch == "submit":
                result = submit(
                    Calculation,
                    code=code,
                    struct=structure,
                    metadata=metadata,
                    config=conf,
                    model=model,
                )
                list_of_nodes.append(result.pk)

                group.add_nodes(load_node(result.pk))

                print(f"Printing results from calculation: {result}")

    print(f"printing dictionary with all {list_of_nodes}")
    # write list of nodes in csv file
    # Unnecessary but might be useful. better use group to query
    with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "PK"])
        for node in list_of_nodes:
            writer.writerow([load_node(node).label, node])


@click.command("cli")
@click.option("--folder", type=Path)
@click.option("--config", type=Path, help="Config file to use")
@click.option("--calc", type=str, help="Calc to run", default="sp")
@click.option("--output_filename", type=str, default="list_nodes.csv")
@click.option("--codelabel", type=str)
@click.option("--group", type=int)
@click.option(
    "--launch", type=str, default="submit", help="can be run_get_pk or submit"
)
# pylint: disable=too-many-arguments
def cli(folder, config, calc, output_filename, codelabel, group, launch):
    """Click interface."""
    try:
        code = load_code(codelabel)
    except NotExistent:
        print(f"The code '{codelabel}' does not exist.")
        sys.exit(1)
    try:
        group = load_group(group)
    except NotExistent:
        print(f"The group '{group}' does not exist.")

    run_ht(folder, config, calc, output_filename, code, group, launch)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
