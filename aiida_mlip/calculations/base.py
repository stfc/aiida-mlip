"""Base class for features common to most calculations."""

from ase.io import read, write

from aiida.common import InputValidationError, datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import SinglefileData, Str, StructureData

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData


def validate_inputs(
    inputs: dict, port_namespace: aiida.engine.processes.ports.PortNamespace
):
    """
    Check if the inputs are valid.

    Parameters
    ----------
    inputs : dict
        The inputs dictionary.

    port_namespace : `aiida.engine.processes.ports.PortNamespace`
        An instance of aiida's `PortNameSpace`.

    Raises
    ------
    ValueError
        Error message if validation fails, None otherwise.
    """
    if "struct" in port_namespace:
        if "struct" not in inputs and "config" not in inputs:
            raise InputValidationError(
                "Either 'struct' or 'config' must be specified in the inputs"
            )
        if (
            "struct" not in inputs
            and "config" in inputs
            and "struct" not in inputs["config"]
        ):
            raise InputValidationError(
                "Structure must be specified through 'struct' or 'config'"
            )


class BaseJanus(CalcJob):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run single point calculations using mlips.

    Attributes
    ----------
    DEFAULT_OUTPUT_FILE : str
        Default stdout file name.
    DEFAULT_INPUT_FILE : str
        Default input file name.
    LOG_FILE : str
        Default log file name.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    validate_inputs(value: dict, port_namespace: PortNamespace) -> Optional[str]:
        Check if the inputs are valid.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

    DEFAULT_OUTPUT_FILE = "aiida-stdout.txt"
    DEFAULT_INPUT_FILE = "aiida.xyz"
    LOG_FILE = "aiida.log"

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
        spec.inputs.validator = validate_inputs
        # Define inputs
        spec.input(
            "arch",
            valid_type=Str,
            required=False,
            help="Mlip architecture to use for calculation, defaults to mace",
        )
        spec.input(
            "model",
            valid_type=ModelData,
            required=False,
            help="Mlip model used for calculation",
        )
        spec.input(
            "struct",
            valid_type=StructureData,
            required=False,
            help="The input structure.",
        )
        spec.input(
            "precision",
            valid_type=Str,
            required=False,
            help="Precision level for calculation",
        )
        spec.input(
            "device",
            valid_type=Str,
            required=False,
            help="Device on which to run calculation (cpu, cuda or mps)",
        )

        spec.input(
            "log_filename",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls.LOG_FILE),
            help="Name of the log output file",
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls.DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "metadata.options.input_filename",
            valid_type=str,
            default=cls.DEFAULT_INPUT_FILE,
        )
        spec.input(
            "metadata.options.scheduler_stdout",
            valid_type=str,
            default="_scheduler-stdout.txt",
            help="Filename to which the content of stdout of the scheduler is written.",
        )

        spec.input(
            "config",
            valid_type=JanusConfigfile,
            required=False,
            help="Name of the log output file",
        )

        spec.output("std_output", valid_type=SinglefileData)
        spec.output("log_output", valid_type=SinglefileData)
        # Exit codes
        spec.exit_code(
            305,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Some output files missing or cannot be read",
        )

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

        # Create needed inputs

        if "struct" in self.inputs:
            structure = self.inputs.struct
        elif "config" in self.inputs and "struct" in self.inputs.config.as_dictionary:
            structure = StructureData(
                ase=read(self.inputs.config.as_dictionary["struct"])
            ).store()

        # Transform the structure data in xyz file called input_filename
        input_filename = self.inputs.metadata.options.input_filename
        atoms = structure.get_ase()
        # with folder.open(input_filename, mode="w", encoding='utf8') as file:
        write(folder.abspath + "/" + input_filename, images=atoms)

        log_filename = (self.inputs.log_filename).value
        cmd_line = {
            "struct": input_filename,
            "log": log_filename,
        }

        # The inputs are saved in the node, but we want their value as a string
        if "precision" in self.inputs:
            precision = (self.inputs.precision).value
            cmd_line["calc-kwargs"] = {"default_dtype": precision}
        if "device" in self.inputs:
            device = (self.inputs.device).value
            cmd_line["device"] = device

        # Define architecture from model if model is given,
        # otherwise get architecture from inputs and download default model
        architecture = None
        architecture = (
            str((self.inputs.model).architecture)
            if "model" in self.inputs and hasattr(self.inputs.model, "architecture")
            else str(self.inputs.arch.value) if "arch" in self.inputs else None
        )

        if architecture:
            cmd_line["arch"] = architecture

        model_path = None
        if "model" in self.inputs:
            model_path = self.inputs.model.filepath
        else:
            if "config" in self.inputs and "model" in self.inputs.config:
                model_path = None
            else:
                if "arch" in self.inputs:
                    model_path = ModelData.download(
                        "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",  # pylint: disable=line-too-long
                        architecture,
                    ).filepath
        if model_path:
            cmd_line.setdefault("calc-kwargs", {})["model"] = model_path

        if "config" in self.inputs:
            # Check if there are values in the config file that are also in the command
            # line and do not store them as only the cmd line parameters will be used
            config_dict = self.inputs.config.as_dictionary
            overlapping_params = cmd_line.keys() & config_dict.keys()
            # Store the other parameters
            self.inputs.config.store_content(skip=overlapping_params)
            # Add config file to command line
            cmd_line["config"] = "config.yaml"
            config_parse = self.inputs.config.get_content()
            # Copy config file content inside the folder where the calculation is run
            with folder.open("config.yaml", "w", encoding="utf-8") as configfile:
                configfile.write(config_parse)

        codeinfo = datastructures.CodeInfo()

        # Initialize cmdline_params with a placeholder "calculation" command
        codeinfo.cmdline_params = ["calculation"]

        for flag, value in cmd_line.items():
            codeinfo.cmdline_params += [f"--{flag}", str(value)]

        # Node where the code is saved
        codeinfo.code_uuid = self.inputs.code.uuid
        # Save name of output as you need it for running the code
        codeinfo.stdout_name = self.metadata.options.output_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # Save the info about the node where the calc is stored
        calcinfo.uuid = str(self.uuid)
        # Retrieve output files
        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
            self.uuid,
            log_filename,
        ]

        return calcinfo
