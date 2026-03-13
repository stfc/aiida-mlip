"""Parsers provided by aiida_mlip. The parser is based on the sp_parser.py written by Ben Speake"""

from __future__ import annotations

from pathlib import Path

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

import yaml
import h5py
import os
import numpy as np


from aiida_mlip.helpers.converters import convert_numpy
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
            print(PhononCalc, node.process_class)
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

        xyz_output = (self.node.inputs.out).value
        nohdf5 = (self.node.inputs.no_hdf5).value
        dos = (self.node.inputs.dos).value

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {xyz_output}
        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Add output file to the outputs
        self.logger.info(f"Parsing '{xyz_output}'")
        
        with self.retrieved.open(xyz_output, "rb") as handle:
            self.out("xyz_output", SinglefileData(file=handle, filename=xyz_output))

        content = None
        with open(Path(self.node.get_remote_workdir(), xyz_output)) as f:
            content = yaml.safe_load(f)
        
        #print("Content read from file:", content)
        results_node = Dict(content)
        self.out("results_dict", results_node)

        #dos
        if dos:
            content = None
            tmp_path = os.path.join(Path(self.node.get_remote_workdir(), "aiida-dos.dat"))
            retrieved = self.retrieved

            try:
                filepath = retrieved.base.repository.get_object_content( "aiida-dos.dat", mode='rb')
            except Exception:
                print("exception in getting filepath")
                return self.exit_codes.ERROR_MISSING_OUTPUT
            
            with open(tmp_path, 'wb') as handle:
                handle.write(filepath)

            results_node = SinglefileData(file=tmp_path)
            self.out("dos", results_node)

        if nohdf5 == False:
            fc_output = "aiida-force_constants.hdf5"
            retrieved = self.retrieved

            try:
                filepath = retrieved.base.repository.get_object_content(fc_output, mode='rb')
            except Exception:
                print("exception in getting filepath")
                return self.exit_codes.ERROR_MISSING_OUTPUT

        # Write temporary file
           
            tmp_path = os.path.join(Path(self.node.get_remote_workdir(), fc_output))
            

            with open(tmp_path, 'wb') as handle:
                handle.write(filepath)

                hdf5_node = SinglefileData(file=tmp_path)
                
            self.out('force_constant', hdf5_node)

            
        return ExitCode(0)
