"""Example code for submitting single point calculation"""
import click
from aiida.engine import run_get_node, submit, run, run_get_pk
from aiida.orm import load_code, load_node, load_group
from aiida.plugins import CalculationFactory
from pathlib import Path
from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure
import csv
import sys
from aiida.common import NotExistent
import time

def run_hts(folder,config,calc, output_filename,code,group,launch):
    # Add the required inputs for aiida
    metadata = {"options": {"resources": {"num_machines": 1}}}
    
    # All the other paramenters we want them from the config file
    # We want to pass it as a AiiDA data type for the provenance
    conf = JanusConfigfile(config)
    # Define calculation to run
    Calculation = CalculationFactory(f"mlip.{calc}")
    list_of_nodes = []
    p = Path(folder)
    for child in p.glob('**/*'):
        if child.name.endswith("cif"):   
            print(child.name)
            metadata['label']=f"{child.name}"
            # This structure will overwrite the one in the config file if present
            structure = load_structure(child.absolute())
            # Run calculation
            if launch == "run_get_pk":
                result,pk = run_get_pk(
                Calculation,
                code=code,
                struct=structure,
                metadata=metadata,
                config=conf,
            )
                list_of_nodes.append(pk)

                group.add_nodes(load_node(pk))
                time.sleep(1)
                print(f"Printing results from calculation: {result}")
                
            if launch== "submit":
                result = submit(
                    Calculation,
                    code=code,
                    struct=structure,
                    metadata=metadata,
                    config=conf,
                )
                list_of_nodes.append(result.pk)

                group.add_nodes(load_node(result.pk))
                time.sleep(5)

                print(f"Printing results from calculation: {result}")

    print(f"printing dictionary with all {list_of_nodes}")
    # write list of nodes in csv file
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "PK"])
        for node in list_of_nodes:
            writer.writerow([load_node(node).label, node])

@click.command('cli')
@click.option('--folder', type=Path)
@click.option('--config', type=Path, help='Config file to use',default = "/work4/scd/scarf1228/config_janus.yaml")
@click.option('--calc', type=str, help='Calc to run', default="sp")
@click.option('--output_filename', type=str, default="list_nodes.csv")
@click.option('--codelabel',type=str, default="janus@scarf-hq")
@click.option('--group', type=int, default=8)
@click.option('--launch', type=str,default="submit", help="can be run_get_pk or submit")
def cli(folder,config,calc, output_filename,codelabel,group,launch):
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
    
    run_hts(folder,config,calc, output_filename,code,group,launch)

if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
