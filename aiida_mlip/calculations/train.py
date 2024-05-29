"""Class for training machine learning models."""

from pathlib import Path

from aiida.common import InputValidationError, datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Bool, Dict, FolderData, SinglefileData

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
    InputValidationError
        Error message if validation fails, None otherwise.
    """
    if "mlip_config" in port_namespace:
        # Check if a config file is given
        if "mlip_config" not in inputs:
            raise InputValidationError("No config file given")
        config_file = inputs["mlip_config"]
        # Check if 'name' keyword is given
        if "name" not in config_file:
            raise InputValidationError("key 'name' must be defined in the config file")
        # Check if the xyz files paths are given
        required_keys = ("train_file", "valid_file", "test_file")
        for key in required_keys:
            if key not in config_file:
                raise InputValidationError(f"Mandatory key {key} not in config file")
            # Check if the keys actually correspond to a path
            if not ((Path(config_file.as_dictionary[key])).resolve()).exists():
                raise InputValidationError(f"Path given for {key} does not exist")
        # Check if fine-tuning is enabled and validate accordingly
        if (
            inputs["fine_tune"]
            and "foundation_model" not in config_file
            and "foundation_model" not in inputs
        ):
            raise InputValidationError(
                "Undefined Model to fine-tune in inputs or config file"
            )


class Train(CalcJob):  # numpydoc ignore=PR01
    """
    Calcjob implementation to train mlips.

    Attributes
    ----------
    DEFAULT_OUTPUT_FILE : str
        Default stdout file name.

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
            "mlip_config",
            valid_type=JanusConfigfile,
            required=True,
            help="Config file with parameters for training",
        )

        spec.input(
            "fine_tune",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help="Whether fine-tuning a model",
        )
        spec.input(
            "foundation_model",
            valid_type=ModelData,
            required=False,
            help="Model to fine-tune",
        )

        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls.DEFAULT_OUTPUT_FILE,
        )

        spec.input(
            "metadata.options.scheduler_stdout",
            valid_type=str,
            default="_scheduler-stdout.txt",
            help="Filename to which the content of stdout of the scheduler is written.",
        )
        spec.inputs["metadata"]["options"]["parser_name"].default = "mlip.train_parser"
        spec.inputs.validator = validate_inputs
        spec.output("model", valid_type=ModelData)
        spec.output("compiled_model", valid_type=SinglefileData)
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the training.",
        )
        spec.output("logs", valid_type=FolderData)
        spec.output("checkpoints", valid_type=FolderData)
        spec.default_output_node = "results_dict"
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
        # The config file needs to be copied in the working folder
        # Read content
        mlip_dict = self.inputs.mlip_config.as_dictionary
        config_parse = self.inputs.mlip_config.get_content()

        # Extract paths from the config
        for file in ("train_file", "test_file", "valid_file"):
            abs_path = Path(mlip_dict[file]).resolve()

            # Update the config file with absolute paths
            config_parse = config_parse.replace(mlip_dict[file], str(abs_path))

        # Add foundation_model to the config file if fine-tuning is enabled
        if self.inputs.fine_tune and "foundation_model" in self.inputs:
            model_data = self.inputs.foundation_model
            foundation_model_path = model_data.filepath
            config_parse += f"\nfoundation_model: {foundation_model_path}"

        # Copy config file content inside the folder where the calculation is run
        config_copy = "mlip_train.yml"
        with folder.open(config_copy, "w", encoding="utf-8") as configfile:
            configfile.write(config_parse)

        codeinfo = datastructures.CodeInfo()

        # Initialize cmdline_params with train command
        codeinfo.cmdline_params = ["train"]
        # Create the rest of the command line
        cmd_line = {"mlip-config": config_copy}
        if self.inputs.fine_tune:
            cmd_line["fine-tune"] = None

        # Add cmd line params to codeinfo
        for flag, value in cmd_line.items():
            if value is None:
                codeinfo.cmdline_params += [f"--{flag}"]
            else:
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
        model_dir = Path(mlip_dict.get("model_dir", "."))
        model_output = model_dir / f"{mlip_dict['name']}.model"
        compiled_model_output = model_dir / f"{mlip_dict['name']}_compiled.model"
        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
            self.uuid,
            mlip_dict.get("log_dir", "logs"),
            mlip_dict.get("result_dir", "results"),
            mlip_dict.get("checkpoint_dir", "checkpoints"),
            str(model_output),
            str(compiled_model_output),
        ]

        return calcinfo
