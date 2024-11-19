"""Class to run descriptors calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Bool, Dict, SinglefileData, Str

from aiida_mlip.calculations.base import BaseJanus


class Descriptors(BaseJanus):  # numpydoc ignore=PR01
    """
    Calcjob implementation to calculate MLIP descriptors.

    Attributes
    ----------
    XYZ_OUTPUT : str
        Default xyz output file name.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    validate_inputs(value: dict, port_namespace: PortNamespace) -> Optional[str]:
        Check if the inputs are valid.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

    XYZ_OUTPUT = "aiida-results.xyz"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the process specification, its inputs, outputs and exit codes.

        Parameters
        ----------
        spec : `aiida.engine.CalcJobProcessSpec`
            The calculation job process spec to define.
        """
        super().define(spec)

        # Define inputs

        spec.input(
            "out",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls.XYZ_OUTPUT),
            help="Name of the xyz output file",
        )

        spec.input(
            "invariants_only",
            valid_type=Bool,
            required=False,
            help="Only calculate invariant descriptors.",
        )

        spec.input(
            "calc_per_element",
            valid_type=Bool,
            required=False,
            help="Calculate mean descriptors for each element.",
        )

        spec.input(
            "calc_per_atom",
            valid_type=Bool,
            required=False,
            help="Calculate descriptors for each atom.",
        )

        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "mlip.descriptors_parser"

        # Define outputs. The default is an array with the descriptors from the xyz file
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `descriptors` output node of the successful calculation.",
        )
        spec.output("xyz_output", valid_type=SinglefileData)

        print("defining outputnode")
        spec.default_output_node = "results_dict"

    # pylint: disable=too-many-locals
    def prepare_for_submission(
        self, folder: aiida.common.folders.Folder
    ) -> datastructures.CalcInfo:
        """
        Create the input files for the `Calcjob`.

        Parameters
        ----------
        folder : aiida.common.folders.Folder
            Folder where the calculation is run.

        Returns
        -------
        aiida.common.datastructures.CalcInfo
            An instance of `aiida.common.datastructures.CalcInfo`.
        """
        # Call the parent class method to prepare common inputs
        calcinfo = super().prepare_for_submission(folder)
        codeinfo = calcinfo.codes_info[0]

        # Adding command line params for when we run janus
        # descriptors is overwriting the placeholder "calculation" from the base.py file
        codeinfo.cmdline_params[0] = "descriptors"

        # The inputs are saved in the node, but we want their value as a string
        xyz_filename = (self.inputs.out).value
        codeinfo.cmdline_params += ["--out", xyz_filename]

        cmdline_options = {}
        if "invariants_only" in self.inputs:
            cmdline_options["invariants-only"] = self.inputs.invariants_only.value
        if "calc_per_element" in self.inputs:
            cmdline_options["calc-per-element"] = self.inputs.calc_per_element.value
        if "calc_per_atom" in self.inputs:
            cmdline_options["calc-per-atom"] = self.inputs.calc_per_atom.value

        for flag, value in cmdline_options.items():
            if isinstance(value, bool):
                # Add boolean flags without value if True
                if value:
                    codeinfo.cmdline_params.append(f"--{flag}")
            else:
                codeinfo.cmdline_params += [f"--{flag}", value]

        calcinfo.retrieve_list.append(xyz_filename)

        return calcinfo
