"""
Parser for mlip train.
"""

import json
from pathlib import Path
from typing import Any

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
        Initialize the TrainParser instance.

    parse(**kwargs: Any) -> int:
        Parse outputs, store results in the database.

    _get_remote_dirs(mlip_dict: [str, Any]) -> [str, Path]:
        Get the remote directories based on mlip config file.

    _validate_retrieved_files(output_filename: str, model_name: str) -> bool:
        Validate that the expected files have been retrieved.

    _save_models(model_output: Path, compiled_model_output: Path) -> None:
        Save model and compiled model as outputs.

    _parse_results(result_name: Path) -> None:
        Parse the results file and store the results dictionary.

    _save_folders(remote_dirs: [str, Path]) -> None:
        Save log and checkpoint folders as outputs.

    Returns
    -------
    int
        An exit code.

    Raises
    ------
    exceptions.ParsingError
        If the ProcessNode being passed was not produced by a `Train` Calcjob.
    """

    def __init__(self, node: ProcessNode):
        """
        Initialize the TrainParser instance.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)

    def parse(self, **kwargs: Any) -> int:
        """
        Parse outputs and store results in the database.

        Parameters
        ----------
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        int
            An exit code.
        """
        mlip_dict = self.node.inputs.mlip_config.as_dictionary
        output_filename = self.node.get_option("output_filename")
        remote_dirs = self._get_remote_dirs(mlip_dict)

        model_output = remote_dirs["model"] / f"{mlip_dict['name']}.model"
        compiled_model_output = (
            remote_dirs["model"] / f"{mlip_dict['name']}_compiled.model"
        )
        result_name = remote_dirs["results"] / f"{mlip_dict['name']}_run-123_train.txt"

        if not self._validate_retrieved_files(output_filename, mlip_dict["name"]):
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self._save_models(model_output, compiled_model_output)
        self._parse_results(result_name)
        self._save_folders(remote_dirs)

        return ExitCode(0)

    def _get_remote_dirs(self, mlip_dict: dict) -> dict:
        """
        Get the remote directories based on mlip config file.

        Parameters
        ----------
        mlip_dict : dict
            Dictionary containing mlip config file.

        Returns
        -------
        dict
            Dictionary of remote directories.
        """
        rem_dir = Path(self.node.get_remote_workdir())
        return {
            typ: rem_dir / mlip_dict.get(f"{typ}_dir", default)
            for typ, default in (
                ("log", "logs"),
                ("checkpoint", "checkpoints"),
                ("results", "results"),
                ("model", ""),
            )
        }

    def _validate_retrieved_files(self, output_filename: str, model_name: str) -> bool:
        """
        Validate that the expected files have been retrieved.

        Parameters
        ----------
        output_filename : str
            The expected output filename.
        model_name : str
            The name of the model as found in the config file key `name`.

        Returns
        -------
        bool
            True if the expected files are retrieved, False otherwise.
        """
        files_retrieved = self.retrieved.list_object_names()
        files_expected = {output_filename, f"{model_name}.model"}

        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return False
        return True

    def _save_models(self, model_output: Path, compiled_model_output: Path) -> None:
        """
        Save model and compiled model as outputs.

        Parameters
        ----------
        model_output : Path
            Path to the model output file.
        compiled_model_output : Path
            Path to the compiled model output file.
        """
        architecture = "mace_mp"
        model = ModelData.local_file(model_output, architecture=architecture)
        compiled_model = ModelData.local_file(
            compiled_model_output, architecture=architecture
        )

        self.out("model", model)
        self.out("compiled_model", compiled_model)

    def _parse_results(self, result_name: Path) -> None:
        """
        Parse the results file and store the results dictionary.

        Parameters
        ----------
        result_name : Path
            Path to the result file.
        """
        with open(result_name, encoding="utf-8") as file:
            last_dict_str = None
            for line in file:
                try:
                    last_dict_str = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

        if last_dict_str is not None:
            results_node = Dict(last_dict_str)
            self.out("results_dict", results_node)
        else:
            raise ValueError("No valid dictionary in the file")

    def _save_folders(self, remote_dirs: dict) -> None:
        """
        Save log and checkpoint folders as outputs.

        Parameters
        ----------
        remote_dirs : dict
            Dictionary of remote folders.
        """
        log_node = FolderData(tree=remote_dirs["log"])
        self.out("logs", log_node)

        checkpoint_node = FolderData(tree=remote_dirs["checkpoint"])
        self.out("checkpoints", checkpoint_node)
