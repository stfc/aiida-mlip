"""Class to run single point calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Dict, SinglefileData, Str

from aiida_mlip.calculations.base import BaseJanus


class Singlepoint(BaseJanus):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run single point calculations using mlips.

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
            "properties",
            valid_type=Str,
            required=False,
            help="Properties to calculate",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "janus.sp_parser"

        # Define outputs. The default is a dictionary with the content of the xyz file
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
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
            An `aiida.common.folders.Folder` to temporarily write files on disk.

        Returns
        -------
        aiida.common.datastructures.CalcInfo
            An instance of `aiida.common.datastructures.CalcInfo`.
        """
        # Call the parent class method to prepare common inputs
        calcinfo = super().prepare_for_submission(folder)
        codeinfo = calcinfo.codes_info[0]

        # Adding command line params for when we run janus
        # singlepoint is overwriting the placeholder "calculation" from the base.py file
        codeinfo.cmdline_params[0] = "singlepoint"

        # The inputs are saved in the node, but we want their value as a string
        xyz_filename = (self.inputs.out).value
        codeinfo.cmdline_params += ["--out", xyz_filename]

        if "properties" in self.inputs:
            properties = self.inputs.properties.value
            codeinfo.cmdline_params += ["--properties", properties]

        calcinfo.retrieve_list.append(xyz_filename)

        return calcinfo
