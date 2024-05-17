"""
Parser for mlip train.
"""

import ast
from pathlib import Path

from aiida.engine import ExitCode
from aiida.orm import Dict, FolderData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.parsers.parser import Parser

from aiida_mlip.data.model import ModelData


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
        print(self.node.inputs.mlip_config)
        mlip_dict = self.node.inputs.mlip_config.as_dictionary
        remote_dirs = {
            typ: remote_dir / mlip_dict.get(f"{typ}_dir", default)
            for typ, default in (
                ("log", "logs"),
                ("checkpoint", "checkpoints"),
                ("results", "results"),
                ("model", ""),
            )
        }

        output_filename = self.node.get_option("output_filename")
        model_output = remote_dirs["model"] / f"{mlip_dict['name']}.model"
        compiled_model_output = (
            remote_dirs["model"] / f"{mlip_dict['name']}_compiled.model"
        )
        result_name = remote_dirs["results"] / f"{mlip_dict['name']}_run-2024_train.txt"

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {output_filename}
        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Save models as outputs
        # Need to change the architecture
        architecture = "mace_mp"
        model = ModelData.local_file(model_output, architecture=architecture)
        compiled_model = ModelData.local_file(
            compiled_model_output, architecture=architecture
        )
        self.out("model", model)
        self.out("compiled_model", compiled_model)

        # In the result file find the last dictionary
        with open(result_name, encoding="utf-8") as file:
            last_dict_str = None
            for line in file:
                try:
                    last_dict_str = ast.literal_eval(line.strip())
                except (SyntaxError, ValueError):
                    continue

        # Convert the last dictionary string to a Dict
        if last_dict_str is not None:
            results_node = Dict(last_dict_str)
            self.out("results_dict", results_node)
        else:
            raise ValueError("No valid dictionary in the file")

        # Save log folder as output
        log_node = FolderData(tree=remote_dirs["log"])
        self.out("logs", log_node)

        # Save checkpoint folder as output
        checkpoint_node = FolderData(tree=remote_dirs["checkpoint"])
        self.out("checkpoints", checkpoint_node)

        return ExitCode(0)
