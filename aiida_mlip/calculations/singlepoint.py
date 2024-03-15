"""Class to run single point calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Dict, SinglefileData, Str, StructureData

from aiida_mlip.data.model import ModelData


class Singlepoint(CalcJob):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run single point calculations using mlips.

    Attributes
    ----------
    _DEFAULT_OUTPUT_FILE : str
        Default stdout file name.
    _DEFAULT_INPUT_FILE : str
        Default input file name.
    _XYZ_OUTPUT : str
        Default xyz output file name.
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
    _DEFAULT_INPUT_FILE = "aiida.cif"
    _XYZ_OUTPUT = "aiida-results.xyz"
    _LOG_FILE = "aiida.log"

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
            "calctype",
            valid_type=Str,
            default=lambda: Str("singlepoint"),
            help="calculation type (single point or geom opt)",
        )
        spec.input(
            "architecture",
            valid_type=Str,
            default=lambda: Str("mace_mp"),
            help="Architecture to use for calculation, defaults to mace_mp",
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
            help="Device on which to run calculation (cpu, cuda or mps)",
        )

        spec.input(
            "xyz_output_name",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls._XYZ_OUTPUT),
            help="Name of the xyz output file",
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
        spec.inputs["metadata"]["options"]["parser_name"].default = "janus.parser"
        spec.inputs.validator = cls.validate_inputs
        # Define outputs. The default is a dictionary with the content of the xyz file
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
        )
        print("defining outputs")
        spec.output("std_output", valid_type=SinglefileData)
        spec.output("log_output", valid_type=SinglefileData)
        spec.output("xyz_output", valid_type=SinglefileData)
        print("defining outputnode")
        spec.default_output_node = "results_dict"

        # Exit codes

        spec.exit_code(
            305,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Some output files missing or cannot be read",
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
        if any(key not in port_namespace for key in ("calctype", "structure")):
            raise ValueError("Both 'calctype' and 'structure' namespaces are required.")

        for key in ("calctype", "structure"):
            if key not in inputs:
                raise ValueError(
                    f"Required value was not provided for the `{key}` namespace."
                )

        valid_calctypes = {"singlepoint", "geom opt"}
        if (
            "calctype" in inputs
            and str(inputs["calctype"].value) not in valid_calctypes
        ):
            raise ValueError(
                f"The 'calctype' must be one of {valid_calctypes}, "
                f"but got '{inputs['calctype']}'."
            )

        if "input_filename" in inputs:
            if not inputs["input_filename"].value.endswith(".cif"):
                raise ValueError("The parameter 'input_filename' must end with '.cif'")

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
            if self.inputs.model
            else str(self.inputs.architecture.value)
        )
        if self.inputs.model:
            model_path = self.inputs.model.filepath
        else:
            model_path = ModelData.download(
                "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",  # pylint:disable=line-too-long
                architecture,
            ).filepath

        # The inputs are saved in the node, but we want their value as a string
        calctype = (self.inputs.calctype).value
        precision = (self.inputs.precision).value
        device = (self.inputs.device).value
        xyz_filename = (self.inputs.xyz_output_name).value
        input_filename = self.inputs.metadata.options.input_filename
        log_filename = (self.inputs.log_filename).value
        # Transform the structure data in cif file called input_filename
        structure = self.inputs.structure
        cif_structure = structure.get_cif()
        with folder.open(self._DEFAULT_INPUT_FILE, "w", encoding="utf-8") as inputfile:
            inputfile.write(cif_structure.get_content())

        cmd_line = {
            "arch": architecture,
            "struct": input_filename,
            "device": device,
            "log": log_filename,
            "calc-kwargs": {"model": model_path, "default_dtype": precision},
            "write-kwargs": {"filename": xyz_filename},
        }

        codeinfo = datastructures.CodeInfo()

        # Initialize cmdline_params as an empty list
        codeinfo.cmdline_params = []
        # Adding command line params for when we run janus
        codeinfo.cmdline_params.append(calctype)
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
            xyz_filename,
            self.uuid,
            log_filename,
        ]

        return calcinfo
