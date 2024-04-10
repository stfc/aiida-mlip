"""Base class for features common to most calculations."""

from ase.io import write
import yaml

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import SinglefileData, Str, StructureData

from aiida_mlip.data.model import ModelData


class BaseJanus(CalcJob):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run single point calculations using mlips.

    Attributes
    ----------
    _DEFAULT_OUTPUT_FILE : str
        Default stdout file name.
    _DEFAULT_INPUT_FILE : str
        Default input file name.
    _LOG_FILE : str
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

    _DEFAULT_OUTPUT_FILE = "aiida-stdout.txt"
    _DEFAULT_INPUT_FILE = "aiida.xyz"
    _LOG_FILE = "aiida.log"

    @classmethod
    def from_config(cls, config_file):
        """
        Parse config file.

        Parameters
        ----------
        config_file : Filepath
            The config file path.

        Returns
        -------
        dict
            Config parameters loaded as a dictionary.
        """
        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(**config)

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
            "architecture",
            valid_type=Str,
            default=lambda: Str("mace"),
            help="Mlip architecture to use for calculation, defaults to mace",
        )
        spec.input(
            "model",
            valid_type=ModelData,
            required=False,
            help="Mlip model used for calculation",
        )
        spec.input("struct", valid_type=StructureData, help="The input structure.")
        spec.input("precision", valid_type=Str, help="Precision level for calculation")
        spec.input(
            "device",
            valid_type=Str,
            required=False,
            default=lambda: Str("cpu"),
            help="Device on which to run calculation (cpu, cuda or mps)",
        )

        spec.input(
            "log_filename",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls._LOG_FILE),
            help="Name of the log output file",
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "metadata.options.input_filename",
            valid_type=str,
            default=cls._DEFAULT_INPUT_FILE,
        )
        spec.input(
            "metadata.options.scheduler_stdout",
            valid_type=str,
            default="_scheduler-stdout.txt",
            help="Filename to which the content of stdout of the scheduler is written.",
        )
        spec.inputs.validator = cls.validate_inputs

        spec.output("std_output", valid_type=SinglefileData)
        spec.output("log_output", valid_type=SinglefileData)
        # Exit codes
        spec.exit_code(
            305,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Some output files missing or cannot be read",
        )
        spec.input(
            "config",
            valid_type=SinglefileData,
            required=False,
            help="Name of the log output file",
        )

    @classmethod
    def validate_inputs(
        cls, inputs: dict, port_namespace: aiida.engine.processes.ports.PortNamespace
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
        # Wrapping processes may choose to exclude certain input ports
        # If the ports have been excluded, skip the validation.
        if "struct" not in port_namespace:
            raise ValueError("'Structure' namespaces is required.")

        if "input_filename" in inputs:
            if not inputs["input_filename"].value.endswith(".xyz"):
                raise ValueError("The parameter 'input_filename' must end with '.xyz'")

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
        # Define architecture from model if model is given,
        # otherwise get architecture from inputs and download default model
        architecture = (
            str((self.inputs.model).architecture)
            if "model" in self.inputs
            else str(self.inputs.architecture.value)
        )
        if "model" in self.inputs:
            model_path = self.inputs.model.filepath
        else:
            model_path = ModelData.download(
                "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",  # pylint:disable=line-too-long
                architecture,
            ).filepath

        # The inputs are saved in the node, but we want their value as a string
        precision = (self.inputs.precision).value
        device = (self.inputs.device).value
        input_filename = self.inputs.metadata.options.input_filename
        log_filename = (self.inputs.log_filename).value

        # Transform the structure data in xyz file called input_filename
        structure = self.inputs.struct

        atoms = structure.get_ase()
        with folder.open(input_filename, "w", encoding="utf-8") as inputfile:
            write(inputfile, images=atoms)

        cmd_line = {
            "arch": architecture,
            "struct": input_filename,
            "device": device,
            "log": log_filename,
            "calc-kwargs": {"model": model_path, "default_dtype": precision},
        }

        if "config" in self.inputs.get_dict():
            cmd_line.update({"config": "config.yaml"})
            config_parse = self.from_config(self.inputs.config.filepath)
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
