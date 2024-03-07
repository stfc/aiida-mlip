"""Class to run single point calculations"""

from typing import Union

from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Dict, SinglefileData, Str, StructureData
from aiida.plugins import ParserFactory

from aiida_mlip.data.model import ModelData


class Singlepoint(CalcJob):
    "Calcjob implementation to run single point calculations using mlips"

    _DEFAULT_INPUT_FILE = "aiida.cif"
    _DEFAULT_OUTPUT_FILE = "aiida.log"
    _XYZ_OUTPUT = Str("aiida.xyz")

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the process specification, including its inputs, outputs and known exit codes.

        :param spec: the calculation job process spec to define.
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
            valid_type=Union[Str, str],
            required=False,
            default=cls._DEFAULT_INPUT_FILE,
            help="Device in which to run calculation(cpu, gpu...)",
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
        # cls.validate_inputs

        # Outputs, in this case it would just be a dictionary with energy etc
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
        )

        spec.output(cls._DEFAULT_OUTPUT_FILE, valid_type=SinglefileData)

        spec.default_output_node = "results_dict"
        # Input errors
        spec.exit_code(300, "INPUT_ERROR", message="Some problems reading the input")
        # Warnings

        # Calculation errors, unrecoverable
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
    def validate_inputs(cls, value):  # Need to check better value
        """Check if the inputs are valid"""

        # Check if model and structure were given as they are required
        for key in "structure":
            if key not in value:
                return f"required value was not provided for the `{key}` namespace."

        return None

    def prepare_for_submission(self, folder):
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
        # Define architecture from model if model is given
        if self.inputs.model:
            model_path = str((self.inputs.model).filepath)
            architecture = str((self.inputs.model).architecture)
        else:
            architecture = str((self.inputs.architecture).value)
            model = ModelData.download("http://tinyurl.com/46jrkm3v", architecture)
            model_path = model.filepath

        # The inputs are saved in the node, but we want their value as a string
        calctype = str((self.inputs.calctype).value)
        # pylint: disable=unused-variable
        precision = str((self.inputs.precision).value)
        device = str((self.inputs.device).value)
        xyz_filename = self.inputs.xyzoutput
        input_filename = self.inputs.metadata.options.input_filename

        # Transform the structure data in cif file called input_filename
        structure = self.inputs.structure
        cif_structure = structure.get_cif()
        with folder.open(
            self._DEFAULT_INPUT_FILE, "w"
        ) as inputfile:  # check better how the folder thing works
            inputfile.write(cif_structure.get_content())

        # Fix kwargs
        cmd_line = {
            "arch": str(architecture),
            "structure": str(input_filename),
            "device": str(device),
            "calc_kwargs": {"model": str(model_path), "default_dtype": str(precision)},
            "write_kwargs": {"filename": str(xyz_filename)},
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
        calcinfo.retrieve_list.append(self.inputs.xyzoutput)
        calcinfo.retrieve_list.append(self.uuid)

        return calcinfo
