"""Base class with inputs for most calculations."""

from ase.io import write

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJob, CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Dict, SinglefileData, Str, StructureData

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