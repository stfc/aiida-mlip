"""Parsers provided by aiida_mlip. The parser is based on sp_parser.py."""

from __future__ import annotations

from pathlib import Path

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory
import yaml

from aiida_mlip.parsers.base_parser import BaseParser

PhononCalc = CalculationFactory("mlip.ph")


class PhononParser(BaseParser):
    """
    Parser class for parsing output of calculation-adapted to accommodate phonons.

    Parameters
    ----------
    node : aiida.orm.nodes.process.process.ProcessNode
        ProcessNode of calculation.

    Methods
    -------
    __init__(node: aiida.orm.nodes.process.process.ProcessNode)
        Initialize the PhononParser instance.

    parse(**kwargs: Any) -> int:
        Parse outputs, store results in the database.

    Returns
    -------
    int
        An exit code.

    Raises
    ------
    exceptions.ParsingError
        If the ProcessNode being passed was not produced by a SinglepointCalc.
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

        if not issubclass(node.process_class, PhononCalc):
            raise exceptions.ParsingError("Can only parse `PhononCalc` calculations")

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
        exit_code = super().parse(**kwargs)

        if exit_code != ExitCode(0):
            return exit_code

        phonon_output = (self.node.inputs.out).value
        nohdf5 = (self.node.inputs.no_hdf5).value
        dos = (self.node.inputs.dos).value
        pdos = (self.node.inputs.pdos).value
        bands = (self.node.inputs.bands).value

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {phonon_output}
        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Add output file to the outputs
        self.logger.info(f"Parsing '{phonon_output}'")

        with self.retrieved.open(phonon_output, "rb") as handle:
            self.out(
                "phonon_output", SinglefileData(file=handle, filename=phonon_output)
            )

        remote_workdir = self.node.get_remote_workdir()
        with Path(remote_workdir, phonon_output).open(encoding="utf-8") as f:
            content = yaml.safe_load(f)

        results_node = Dict(content)
        self.out("results_dict", results_node)

        # dos
        if dos:
            dos_path = Path(remote_workdir) / "aiida-dos.dat"

            try:
                filedata = self.retrieved.base.repository.get_object_content(
                    "aiida-dos.dat", mode="rb"
                )
            except FileNotFoundError:
                self.logger.error(f"exception in filepath for the density of states")
                return self.exit_codes.ERROR_MISSING_OUTPUT

            dos_path.write_bytes(filedata)

            results_node = SinglefileData(file=dos_path)
            self.out("dos", results_node)

        if pdos:
            pdos_path = Path(remote_workdir) / "aiida-pdos.dat"

            try:
                filedata = self.retrieved.base.repository.get_object_content(
                    "aiida-pdos.dat", mode="rb"
                )
            except FileNotFoundError:
                self.logger.info(f"exception in filepath for the partial density of states")
                return self.exit_codes.ERROR_MISSING_OUTPUT

            pdos_path.write_bytes(filedata)

            results_node = SinglefileData(file=pdos_path)
            self.out("pdos", results_node)

        if not nohdf5:
            try:
                filedata = self.retrieved.base.repository.get_object_content(
                    "aiida-force_constants.hdf5", mode="rb"
                )
            except FileNotFoundError:
                self.logger.info(f"exception in getting force constant filepath")
                return self.exit_codes.ERROR_MISSING_OUTPUT

            hdf5_path = Path(remote_workdir) / "aiida-force_constants.hdf5"

            hdf5_path.write_bytes(filedata)
            hdf5_node = SinglefileData(file=hdf5_path)

            self.out("force_constants", hdf5_node)

        # for band structure the required file is aiida-auto_bands.yml.xz
        # this needs to be changed for hdf5
        if bands:
            bnds_output = "aiida-auto_bands.yml.xz"

            try:
                filedata = self.retrieved.base.repository.get_object_content(
                    bnds_output, mode="rb"
                )
            except FileNotFoundError:
                self.logger.info(f"exception in getting force constant filepath")
                return self.exit_codes.ERROR_MISSING_OUTPUT

            bands_path = Path(remote_workdir) / bnds_output

            bands_node = SinglefileData(file=bands_path)

            self.out("band_structure", bands_node)

        return ExitCode(0)
