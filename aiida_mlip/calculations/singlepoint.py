"""Class to run single point calculations"""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Dict, SinglefileData, Str, StructureData

from aiida_mlip.data.model import ModelData


class Singlepoint(CalcJob):
    "Calcjob implementation to run single point calculations using mlips"

    _DEFAULT_INPUT_FILE = "aiida.cif"
    _DEFAULT_OUTPUT_FILE = "aiida.log"
    _XYZ_OUTPUT = "aiida-results.xyz"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the process specification, including its inputs, outputs and known exit codes.

        Parameters
        ----------
        spec : aiida.engine.CalcJobProcessSpec
            the calculation job process spec to define.
        """
        super().define(spec)

        # Define inputs
        spec.input(
            "calctype",
            valid_type=Str,
            help="calculation type (single point or geom opt)",
        )
        spec.input(
            "architecture",
            valid_type=Str,
            default=lambda: Str("mace_mp"),
            help="Architecture to use for calculation, if use default, it will use the default model too",
        )
        spec.input(
            "model",
            valid_type=ModelData,
            required=False,
            help="mlip model used for calculation",
        )
        spec.input("structure", valid_type=StructureData, help="The input structure.")
        spec.input("precision", valid_type=Str, help="Precision level for calculation")
        spec.input(
            "device",
            valid_type=Str,
            required=False,
            default=lambda: Str("cpu"),
            help="Device in which to run calculation(cpu, gpu...)",
        )

        spec.input(
            "xyzoutput",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls._XYZ_OUTPUT),
            help="Name of the xyz output file",
        )
        # additional arguments?
        spec.input(
            "metadata.options.input_filename",
            valid_type=str,
            default=cls._DEFAULT_INPUT_FILE,
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )

        spec.input(
            "metadata.options.scheduler_stdout",
            valid_type=str,
            default="_scheduler-stdout.txt",
            help="Filename to which the content of stdout of the scheduler is written.",
        )
        spec.inputs["metadata"]["options"]["parser_name"].default = "janus.parser"
        # cls.validate_inputs ---> need to work on this

        # Outputs, in this case it would be a dictionary with energy etc and the output files
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
        )
        print("defining outputs")
        spec.output("log_output", valid_type=SinglefileData)
        spec.output("xyz_output", valid_type=SinglefileData)
        print("defining outputnode")
        spec.default_output_node = "results_dict"

        # Exit codes, some are already in the parser, some need to fix
        spec.exit_code(300, "INPUT_ERROR", message="Some problems reading the input")

        spec.exit_code(
            340,
            "ERROR_OUT_OF_WALLTIME_INTERRUPTED",
            message="The calculation stopped prematurely because it ran out of walltime",
        )
        spec.exit_code(
            305,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Some output files missing or cannot be read",
        )
        spec.exit_code(
            306,
            "ERROR_EMPTY_OUTPUT_FILES",
            message="File.xyz is empty",
        )
        spec.exit_code(
            307,
            "ERROR_CONTENT_OUTPUT_FILES",
            message="The output file does not contain the right content",
        )

    @classmethod
    def validate_inputs(cls, value: dict) -> str:  # Need to work on this
        """Check if the inputs are valid.

        Parameters
        ----------
        value : dict
            The inputs dictionary.

        Returns
        -------
        str or None
            Error message if validation fails, None otherwise.
        """

        # Check if  structure is given as it is required
        for key in "structure":
            if key not in value:
                return f"required value was not provided for the `{key}` namespace."

        return None

    # pylint: disable=too-many-locals
    def prepare_for_submission(
        self, folder: aiida.common.folders.Folder
    ) -> datastructures.CalcInfo:
        """
        Create the input files from the input nodes passed to this instance of the `CalcJob`.

        Parameters
        ----------
        folder : aiida.common.folders.Folder
            An `aiida.common.folders.Folder` to temporarily write files on disk.

        Returns
        -------
        aiida.common.datastructures.CalcInfo
            An instance of `aiida.common.datastructures.CalcInfo`.
        """
        # Input parameters
        # Define architecture from model if model is given, otherwise get architecture from inputs and download default model
        architecture = (
            str((self.inputs.model).architecture)
            if self.inputs.model
            else str(self.inputs.architecture.value)
        )
        model_path = (
            str((self.inputs.model).filepath)
            if self.inputs.model
            else ModelData.download(
                "http://tinyurl.com/46jrkm3v", architecture
            ).filepath
        )

        # The inputs are saved in the node, but we want their value as a string
        calctype = str((self.inputs.calctype).value)
        precision = str((self.inputs.precision).value)
        device = str((self.inputs.device).value)
        xyz_filename = str((self.inputs.xyzoutput).value)
        input_filename = self.inputs.metadata.options.input_filename

        # Transform the structure data in cif file called input_filename
        structure = self.inputs.structure
        cif_structure = structure.get_cif()
        with folder.open(
            self._DEFAULT_INPUT_FILE, "w"
        ) as inputfile:  # check better how the folder thing works
            inputfile.write(cif_structure.get_content())

        # Fix kwargs
        # Fix kwargs
        cmd_line = {
            "arch": architecture,
            "structure": str(input_filename),
            "device": device,
            "calc-kwargs": {"model": model_path, "default_dtype": precision},
            "write-kwargs": {"filename": xyz_filename},
        }

        codeinfo = datastructures.CodeInfo()

        # Initialize cmdline_params as an empty list
        codeinfo.cmdline_params = []
        # Adding command line params for when we run janus
        codeinfo.cmdline_params.append(calctype)
        for flag, value in cmd_line.items():
            if isinstance(value, dict):
                # If the value is a dictionary, format it properly
                formatted_value = (
                    "{" + ", ".join([f"'{k}': '{v}'" for k, v in value.items()]) + "}"
                )
                codeinfo.cmdline_params += [f"--{flag}", formatted_value]
            else:
                codeinfo.cmdline_params += [f"--{flag}", str(value)]

        # node where the code is saved
        codeinfo.code_uuid = self.inputs.code.uuid
        # save name of output as you need it for running the code
        codeinfo.stdout_name = self.metadata.options.output_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # Save the info about the node where the calc is stored
        calcinfo.uuid = str(self.uuid)
        # Retrieve by default the output file, need to check about output_filename kw and also input
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self.metadata.options.output_filename)
        calcinfo.retrieve_list.append(xyz_filename)
        calcinfo.retrieve_list.append(self.uuid)

        return calcinfo
