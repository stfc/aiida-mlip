"""Class to run EOS calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
from aiida.orm import Float, Str, SinglefileData
from aiida_mlip.calculations.base import BaseJanus
import aiida.engine.processes
from aiida_mlip.helpers.converters import kwarg_to_param

class EOS(BaseJanus):  # numpydoc ignore=PR01

    XYZ_OUTPUT = "aiida-results.xyz"
    DEFAULT_SUMMARY_FILE = "eos-summary.yml"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:

        super().define(spec)

        spec.input(
            "min_volume",
            valid_type=Float,
            required=False,
            help="Minimum volume scale factor.",
        )

        spec.input(
            "max_volume",
            valid_type=Float,
            required=False,
            help="Maximum volume scale factor.",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "mlip.eos_parser"

        spec.output("xyz_output", valid_type=SinglefileData)


    def prepare_for_submission(
        self, folder: aiida.common.folders.Folder
    ) -> datastructures.CalcInfo:

        calcinfo = super().prepare_for_submission(folder)
        codeinfo = calcinfo.codes_info[0]

        print("outputs", self.inputs)

        cmdline_options = {
            key.replace("_", "-"): getattr(self.inputs, key).value
            for key in ("min_volume", "max_volume", "n_volumes", "eos_type","minimize","minimize_all","fmax", "file_prefix")
            if key in self.inputs
        }
        print(calcinfo)
        
        codeinfo.cmdline_params = [
            "eos",
            "--write-structures",
            *codeinfo.cmdline_params[1:],
            *kwarg_to_param(cmdline_options),
        ]

        calcinfo.retrieve_list.append("xyz_output")
        
        return calcinfo