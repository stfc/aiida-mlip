"""
Parser for mlip train.
"""

from pathlib import Path

from ase.io import read

from aiida.engine import ExitCode
from aiida.orm import Dict, FolderData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.parsers.parser import Parser

from aiida_mlip.data.model import ModelData
from aiida_mlip.helpers.converters import convert_numpy


class TrainParser(Parser):
    """
    Parser class for parsing output of calculation.

    Parameters
    ----------
    node : aiida.orm.nodes.process.process.ProcessNode
        ProcessNode of calculation.

    Methods
    -------
    __init__(node: aiida.orm.nodes.process.process.ProcessNode)
        Initialize the SPParser instance.

    parse(**kwargs: Any) -> int:
        Parse outputs, store results in the database.

    Returns
    -------
    int
        An exit code.

    Raises
    ------
    exceptions.ParsingError
        If the ProcessNode being passed was not produced by a singlePointCalculation.
    """

    def __init__(self, node: ProcessNode):
        """
        Check that the ProcessNode being passed was produced by a `Singlepoint`.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)

    # disable for now
    # pylint: disable=too-many-locals
    def parse(self, **kwargs) -> int:
        """
        Parse outputs, store results in the database.

        Parameters
        ----------
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        int
            An exit code.
        """
        remote_dir = Path(self.node.get_remote_workdir())
        mlip_dict = self.node.get_option("mlip_config").as_dictionary()
        log_dir = remote_dir / Path(mlip_dict.get("log_dir", "logs"))
        checkpoint_dir = remote_dir / Path(
            mlip_dict.get("checkpoint_dir", "checkpoints")
        )
        results_dir = remote_dir / Path(mlip_dict.get("results_dir", "results"))
        model_dir = remote_dir / Path(mlip_dict.get("model_dir", ""))

        output_filename = self.node.get_option("output_filename")
        model_output = model_dir / f"{mlip_dict['name']}.model"
        compiled_model_output = model_dir / f"{mlip_dict['name']}_compiled.model"
        result_name = results_dir / f"{mlip_dict['name']}_run-2024.txt"

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {output_filename}
        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Need to change the architecture
        architecture = "mace_mp"
        model = ModelData.local_file(model_output, architecture=architecture)
        compiled_model = ModelData.local_file(
            compiled_model_output, architecture=architecture
        )
        self.out("model", model)
        self.out("compiled_model", compiled_model)

        content = read(result_name)
        results = convert_numpy(content.todict())
        results_node = Dict(results)
        self.out("results_dict", results_node)

        log_node = FolderData(log_dir)
        self.out("logs", log_node)

        checkpoint_node = FolderData(checkpoint_dir)
        self.out("checkpoints", checkpoint_node)

        return ExitCode(0)
