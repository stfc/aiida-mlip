import csv
from io import BytesIO, StringIO
from pathlib import Path
import re
import time
from typing import Optional, Union

from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction, if_, workfunction
from aiida.orm import (
    Dict,
    Group,
    Int,
    List,
    Node,
    SinglefileData,
    Str,
    StructureData,
    load_code,
    load_group,
    load_node,
)
from aiida.plugins import CalculationFactory, DataFactory

from aiida_mlip.helpers.help_load import load_structure

geomopt = CalculationFactory("mlip.opt")


@calcfunction
def get_input_structures_dict(folder) -> dict[StructureData]:
    struct_dict = {}
    for child in Path(str(folder.value)).glob("**/*.cif"):
        structure = load_structure(child.absolute())
        label = re.sub(r"\W+", "_", child.stem)
        struct_dict.update({label: structure})
    return struct_dict


@calcfunction
def create_csv_file(node_dict: dict, output_filename: str) -> SinglefileData:
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
    return Dict(dict=dictionary)


class HTSWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(geomopt, namespace="calc_inputs", exclude="struct")
        spec.input("folder", valid_type=Str, help="Folder containing CIF files")
        spec.input(
            "launch", valid_type=Str, help='Launch mode: "run_get_pk" or "submit"'
        )
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
            help="The input_structures.",
        )

        spec.output_namespace(
            "output_structures",
            valid_type=StructureData,
            dynamic=True,
            required=False,
            help="The output_structures.",
        )

        spec.expose_outputs(geomopt)
        spec.output("node_dict", valid_type=Dict, help="Dict of calculation nodes")
        # spec.output('energies', valid_type=Dict, help='A dictionary with the energies of all the materials')
        spec.output(
            "csvfile", valid_type=SinglefileData, help="A file with all the outputs"
        )

    def initialize(self):
        # self.ctx.calculation_cls = CalculationFactory(f"{self.inputs.entrypoint.value}")
        self.ctx.folder = Path(self.inputs.folder.value)
        self.ctx.launch = self.inputs.launch.value
        self.ctx.group = load_group(pk=self.inputs.group.value)
        # self.ctx.calcjob_inputs = dict(self.inputs.calc_inputs)
        self.ctx.dict_of_nodes = {}
        self.ctx.successful = []
        self.ctx.failed_runs = []

    def should_run_calculations(self):
        return self.ctx.folder.exists() and any(self.ctx.folder.glob("**/*.cif"))

    def run_calculations(self):
        struct_dict = get_input_structures_dict(self.inputs.folder.value)
        self.out("input_structures", struct_dict)
        inputs = AttributeDict(self.exposed_inputs(geomopt, namespace="calc_inputs"))

        for name, structure in struct_dict.items():
            label = f"{name}"
            inputs["structure"] = structure

            self.report(f"Running calculation for {name}")

            if self.ctx.launch == "run_get_pk":
                future, pk = self.run_get_pk(geomopt, inputs)
                self.report(f"submitting `Geomopt` <PK={pk}>")
                inputs.metadata.label = label
                inputs.metadata.call_link_label = label
                self.to_context(**{label: future})
                time.sleep(self.inputs.settings.sleep_submission_time)

            elif self.ctx.launch == "submit":
                future = self.submit(geomopt, inputs)
                self.report(f"submitting `Geomopt` <PK={future.pk}>")
                inputs.metadata.label = label
                inputs.metadata.call_link_label = label
                self.to_context(**{label: future})
                time.sleep(self.inputs.settings.sleep_submission_time)

    def inspect_all_runs(self):
        """Inspect all previous calculations."""
        outputs_dict = {}
        for label, calculation in self.ctx.items():
            if label.endswith("cif"):
                if calculation.is_finished_ok:
                    outputs_dict[f"{label}"] = calculation.outputs.final_structure
                    self.ctx.dict_of_nodes[f"{label}"] = {
                        "node": calculation.pk,
                        "exit_status": calculation.exit_status,
                        "energy": calculation.outputs.get_dict()["info"]["energy"],
                    }
                    self.ctx.successful.append(calculation.pk)
                    self.ctx.group.add_nodes(pk=calculation.pk)
                else:
                    self.report(
                        f"PwBasecalculation with <PK={calculation.pk}> failed"
                        f"with exit status {calculation.exit_status}"
                    )
                    self.ctx.dict_of_nodes[f"{label}"] = {
                        "node": calculation.pk,
                        "energy": "NaN",
                    }
                    self.ctx.group.add_nodes(pk=calculation.pk)
                    self.ctx.dict_of_nodes.append(calculation.pk)
                    self.ctx.failed_runs.append(calculation.pk)
        self.out("output_structures", outputs_dict)

    def finalize(self):
        self.report(f"Nodes dict: {self.ctx.dict_of_nodes}")
        dict_of_nodes = convert_to_node(self.ctx.dict_of_nodes)
        self.out("node_dict", dict_of_nodes)

        csvfile = create_csv_file(
            self.ctx.dict_of_nodes, self.inputs.output_filename.value
        )
        self.out("csvfile", csvfile)
