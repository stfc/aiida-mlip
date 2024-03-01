"""Class to run single point calculations"""

from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Dict, Str, StructureData

from aiida_mlip.data.model import ModelData


class Singlepoint(CalcJob):
    "Calcjob implementation to run single point calculations using mlips"

    _DEFAULT_OUTPUT_FILE = "aiida.out"
    _DEFAULT_LOG_FILE = "aiida.log"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the process specification, including its inputs, outputs and known exit codes.

        :param spec: the calculation job process spec to define.
        """
        super().define(spec)

        # Define inputs
        spec.input("model", valid_type=ModelData)
        spec.input("structure", valid_type=StructureData, help="The inputs structure.")
        spec.input("precision", valid_type=Str)
        spec.input("device", valid_type=Str)

        spec.inputs.validator = cls.validate_inputs

        # Outputs, in this case it would just be a dictionary with energy etc
        spec.output(
            "output_parameters",
            valid_type=Dict,
            help="The `output_parameters` output node of the successful calculation.",
        )
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
        for key in ("model", "structure"):
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
        # Input parameters that we want to store in the config file
        architecture = self.inputs.model.architecture
        model_path = self.inputs.model.file
        structure = self.inputs.structure
        cif_structure = structure.get_cif()
        device = self.inputs.device

        with folder.open(
            "aiida.cif", "w"
        ) as inputfile:  # check better how the folder thing works
            inputfile.write(cif_structure.get_content())

        cif_file_path = folder / "aiida.cif"

        with folder.open("config", "w") as config:
            config.write(f"architecture:{architecture}")
            config.write(f"model:{model_path}")
            config.write(f"structure:{cif_file_path}")
            config.write(f"device:{device}")

        calcinfo = datastructures.CalcInfo()

        # Save the info about the node where the calc is stored
        calcinfo.uuid = str(self.uuid)

        # Retrieve by default the output file, need to check about output_filename_kw and also input
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self.metadata.options.output_filename)

        return calcinfo
