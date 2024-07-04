"""Workflows to run high-throughput screenings."""

import csv
from io import StringIO
from pathlib import Path
import re
import time

from aiida.common import AttributeDict
from aiida.engine import WorkChain, calcfunction, if_
from aiida.orm import Dict, Int, SinglefileData, Str, StructureData, load_group
from aiida.plugins import CalculationFactory

from aiida_mlip.helpers.help_load import load_structure

geomopt = CalculationFactory("mlip.opt")


@calcfunction
def get_input_structures_dict(folder) -> dict[StructureData]:
    """
    Load CIF files from a folder and create a dictionary of StructureData.

    Parameters
    ----------
    folder : FolderData
        A folder containing CIF files.

    Returns
    -------
    dict
        A dictionary with structure labels as keys and StructureData as values.
    """
    struct_dict = {}
    for child in Path(str(folder.value)).glob("**/*.cif"):
        structure = load_structure(child.absolute())
        label = re.sub(r"\W+", "_", child.stem)
        struct_dict.update({label: structure})
    return struct_dict


@calcfunction
def create_csv_file(node_dict: dict, output_filename: str) -> SinglefileData:
    """
    Create a CSV file from a dictionary of node attributes.

    Parameters
    ----------
    node_dict : dict
        Dictionary containing node attributes.
    output_filename : str
        The name of the output CSV file.

    Returns
    -------
    SinglefileData
        A SinglefileData object containing the CSV file.
    """
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["name", "PK", "energy", "exit status"])
    for nodename, attributes in node_dict.items():
        pk = attributes["node"]
        energy = attributes["energy"]
        exit_status = attributes["exit_status"]
        writer.writerow([nodename, pk, energy, exit_status])
    output.seek(0)
    return SinglefileData(file=output, filename=output_filename)


@calcfunction
def convert_to_node(dictionary):
    """
    Convert a dictionary to an AiiDA Dict node.

    Parameters
    ----------
    dictionary : dict
        The dictionary to convert.

    Returns
    -------
    Dict
        An AiiDA Dict node containing the dictionary.
    """
    return Dict(dict=dictionary)


class HTSWorkChain(WorkChain):
    """
    A high-throughput workflow for running calculations on CIF structures.

    Attributes
    ----------
    ctx : AttributeDict
        Context for storing intermediate data.
    """

    @classmethod
    def define(cls, spec):
        """
        Define the process specification.

        Parameters
        ----------
        spec : ProcessSpec
            The process specification to define inputs, outputs, and  workflow outline.
        """
        super().define(spec)

        spec.input("folder", valid_type=Str, help="Folder containing CIF files")
        spec.input(
            "output_filename",
            valid_type=Str,
            default=Str("outputs.csv"),
            help="Filename for the output CSV",
        )
        spec.input("group", valid_type=Int, help="Group to add the nodes to")
        spec.input("entrypoint", valid_type=Str, help="calculation entry point")
        spec.input(
            "settings.sleep_submission_time",
            valid_type=(int, float),
            non_db=True,
            default=3.0,
            help="Time in seconds to wait before submitting calculations.",
        )
        calc = CalculationFactory(spec.inputs.entrypoint.value)
        spec.expose_inputs(calc, namespace="calc_inputs", exclude="struct")

        spec.outline(
            cls.initialize,
            if_(cls.should_run_calculations)(cls.run_calculations),
            cls.inspect_all_runs,
            cls.finalize,
        )

        spec.output_namespace(
            "input_structures",
            valid_type=StructureData,
            dynamic=True,
            required=False,
            help="The input structures.",
        )

        spec.output_namespace(
            "output_structures",
            valid_type=StructureData,
            dynamic=True,
            required=False,
            help="The output structures.",
        )

        spec.expose_outputs(geomopt)
        spec.output("node_dict", valid_type=Dict, help="Dict of calculation nodes")
        # spec.output('energies', valid_type=Dict, help='dict with the energies')
        spec.output(
            "csvfile", valid_type=SinglefileData, help="A file with all the outputs"
        )

    def initialize(self):
        """Initialize the workchain context."""
        # self.ctx.calculation_cls = CalculationFactory(self.inputs.entrypoint.value)
        self.ctx.folder = Path(self.inputs.folder.value)
        self.ctx.launch = self.inputs.launch.value
        self.ctx.group = load_group(pk=self.inputs.group.value)
        # self.ctx.calcjob_inputs = dict(self.inputs.calc_inputs)
        self.ctx.dict_of_nodes = {}
        self.ctx.successful = []
        self.ctx.failed_runs = []

    def should_run_calculations(self):
        """
        Check if calculations should be run based on the existence of CIF files.

        Returns
        -------
        bool
            True if CIF files exist in the folder, False otherwise.
        """
        return self.ctx.folder.exists() and any(self.ctx.folder.glob("**/*.cif"))

    def run_calculations(self):
        """
        Run calculations for each structure in the input folder.
        """
        struct_dict = get_input_structures_dict(self.inputs.folder.value)
        self.out("input_structures", struct_dict)
        inputs = AttributeDict(self.exposed_inputs(geomopt, namespace="calc_inputs"))

        for name, structure in struct_dict.items():
            label = f"{name}"
            inputs["struct"] = structure

            self.report(f"Running calculation for {name}")

            future = self.submit(geomopt, **inputs)
            self.report(f"submitting `Geomopt` with submit <PK={future.pk}>")
            inputs.metadata.label = label
            inputs.metadata.call_link_label = label
            self.to_context(**{label: future})
            time.sleep(self.inputs.settings.sleep_submission_time)

    def inspect_all_runs(self):
        """
        Inspect all previous calculations and categorize them as successful or failed.
        """
        outputs_dict = {}
        for label, calculation in self.ctx.items():
            if label.endswith("cif"):
                if calculation.is_finished_ok:
                    outputs_dict[f"{label}"] = calculation.outputs.final_structure
                    self.ctx.dict_of_nodes[f"{label}"] = {
                        "node": calculation.pk,
                        "exit_status": calculation.exit_status,
                    }
                    self.ctx.successful.append(calculation.pk)
                    self.ctx.group.add_nodes(pk=calculation.pk)
                else:
                    self.report(
                        f"Calculation with <PK={calculation.pk}> failed"
                        f"with exit status {calculation.exit_status}"
                    )
                    self.ctx.dict_of_nodes[f"{label}"] = {
                        "node": calculation.pk,
                        "exit_status": calculation.exit_status,
                    }
                    self.ctx.group.add_nodes(pk=calculation.pk)
                    self.ctx.dict_of_nodes.append(calculation.pk)
                    self.ctx.failed_runs.append(calculation.pk)
        self.out("output_structures", outputs_dict)

    def finalize(self):
        """
        Finalize the workchain by creating a summary CSV file and output dictionary.
        """
        self.report(f"Nodes dict: {self.ctx.dict_of_nodes}")
        dict_of_nodes = convert_to_node(self.ctx.dict_of_nodes)
        self.out("node_dict", dict_of_nodes)

        csvfile = create_csv_file(
            self.ctx.dict_of_nodes, self.inputs.output_filename.value
        )
        self.out("csvfile", csvfile)
