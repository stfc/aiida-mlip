"""Class for training machine learning models."""

from pathlib import Path

from aiida.common import InputValidationError, datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Dict, FolderData, SinglefileData

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
        if "mlip_config" not in inputs:
            raise InputValidationError("No config file given")
        mlip_dict = inputs.mlip_config.as_dictionary()
        required_keys = ["train_file", "valid_file", "test_file", "name"]
        for key in required_keys:
            if key not in mlip_dict:
                raise InputValidationError(f"Mandatory key {key} not in config file")
            # Check if the keys actually correspond to a path except name which is
            # just the name to use for the output files
            if key != "name":
                if not Path(key).exists():
                    raise InputValidationError(f"Path given for {key} does not exist")


class Train(CalcJob):  # numpydoc ignore=PR01
    """
    Calcjob implementation to train mlips.

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
            "mlip_config",
            valid_type=JanusConfigfile,
            required=False,
            help="Mlip architecture to use for calculation, defaults to mace",
        )
        spec.output("model", valid_type=ModelData)
        spec.output("compiled_model", valid_type=SinglefileData)
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
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
        cmd_line = {}

        cmd_line["mlip-config"] = "mlip_train.yaml"
        config_parse = self.inputs.config.get_content()
        mlip_dict = self.inputs.mlip_config.as_dictionary()
        # Copy config file content inside the folder where the calculation is run
        with folder.open("mlip_config.yaml", "w", encoding="utf-8") as configfile:
            configfile.write(config_parse)

        model_dir = Path(mlip_dict.get("model_dir", "."))
        model_output = model_dir / f"{mlip_dict['name']}.model"
        compiled_model_output = model_dir / f"{mlip_dict['name']}_compiled.model"

        codeinfo = datastructures.CodeInfo()

        # Initialize cmdline_params with train command
        codeinfo.cmdline_params = ["train"]

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
            mlip_dict["log_dir"],
            mlip_dict["result_dir"],
            mlip_dict["checkpoint_dir"],
            model_output,
            compiled_model_output,
        ]

        return calcinfo
