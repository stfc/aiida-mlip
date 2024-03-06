"""Class to run single point calculations"""

from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import SinglefileData, Str, StructureData

from aiida_mlip.data.model import ModelData


class Singlepoint(CalcJob):
    "Calcjob implementation to run single point calculations using mlips"

    _DEFAULT_INPUT_FILE = "aiida.cif"
    _DEFAULT_OUTPUT_FILE = "aiida.xyz"

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
            help="Device where to run calculation(cpu, gpu...)",
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

        # cls.validate_inputs

        # Outputs, in this case it would just be a dictionary with energy etc
        # spec.output(
        #     "output_parameters",
        #     valid_type=Dict,
        #     help="The `output_parameters` output node of the successful calculation.",
        # )
        spec.output(cls._DEFAULT_OUTPUT_FILE, valid_type=SinglefileData)
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
            "ERROR_OUTPUT_FILES",
            message="Some output files missing or cannot be read",
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
            # pylint: disable=unused-variable
            model_path = str((self.inputs.model).filepath)
            architecture = str((self.inputs.model).architecture)
        else:
            architecture = str((self.inputs.architecture).value)
            model_path = None

        # The inputs are saved in the node, but we want their value as a string
        calctype = str((self.inputs.calctype).value)
        # pylint: disable=unused-variable
        precision = str((self.inputs.precision).value)
        device = str((self.inputs.device).value)
        filename = self.inputs.metadata.options.input_filename

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
            # "model": model_path,
            "structure": str(filename),
            "device": str(device),
            # "precision": precision,
        }

        codeinfo = datastructures.CodeInfo()

        # adding command line params for when we run janus
        codeinfo.cmdline_params = [f"{calctype}"]
        for flag in cmd_line.items():
            codeinfo.cmdline_params += [f"--{flag[0]}", f"{flag[1]}"]

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

        return calcinfo
