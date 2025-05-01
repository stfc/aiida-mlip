"""Parsers provided by aiida_mlip."""

from __future__ import annotations

from aiida.common import exceptions
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

from aiida_mlip.parsers.sp_parser import SPParser

DescriptorsCalc = CalculationFactory("mlip.descriptors")


class DescriptorsParser(SPParser):
    """
    Parser class for parsing output of descriptors calculation.

    Inherits from SPParser.

    Parameters
    ----------
    node : aiida.orm.nodes.process.process.ProcessNode
        ProcessNode of calculation.

    Raises
    ------
    exceptions.ParsingError
        If the ProcessNode being passed was not produced by a DescriptorsCalc.
    """

    def __init__(self, node: ProcessNode):
        """
        Check that the ProcessNode being passed was produced by a `Descriptors`.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)

        if not issubclass(node.process_class, DescriptorsCalc):
            raise exceptions.ParsingError("Can only parse `Descriptors` calculations")
