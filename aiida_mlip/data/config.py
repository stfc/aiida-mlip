"""Define Model Data type in AiiDA."""

from pathlib import Path
from typing import Any, Optional, Union

import yaml

from aiida.orm import Data, SinglefileData

from aiida_mlip.helpers.converters import convert_to_nodes


class JanusConfigfile(SinglefileData):
    """
    Define config file type in AiiDA in yaml.

    Parameters
    ----------
    file : Union[str, Path]
        Absolute path to the file.
    filename : Optional[str], optional
        Name to be used for the file (defaults to the name of provided file).

    Attributes
    ----------
    filepath : str
        Path of the config file.

    Methods
    -------
    set_file(file, filename=None, architecture=None, **kwargs)
        Set the file for the node.
    read_yaml()
        Reads the config file from yaml format.
    store_content(store_all: bool = False, skip: list = None) -> dict:
        Converts keys in dictionary to nodes and store them
    as_dictionary(self) -> dict
        Returns the config file as a dictionary.

    Other Parameters
    ----------------
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        file: Union[str, Path],
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ModelData object.

        Parameters
        ----------
        file : Union[str, Path]
            Absolute path to the file.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).

        Other Parameters
        ----------------
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(file, filename, **kwargs)
        self.base.attributes.set("filepath", str(file))

    def __contains__(self, key):
        """
        Check if a key exists in the config file.

        Parameters
        ----------
        key : str
            Key to check.

        Returns
        -------
        bool
            True if the key exists in the config file, False otherwise.
        """
        config = self.as_dictionary
        return key in config

    def set_file(
        self,
        file: Union[str, Path],
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Set the file for the node.

        Parameters
        ----------
        file : Union[str, Path]
            Absolute path to the file.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).

        Other Parameters
        ----------------
        **kwargs : Any
            Additional keyword arguments.
        """
        super().set_file(file, filename, **kwargs)
        self.base.attributes.set("filepath", str(file))

    def read_yaml(self) -> dict:
        """
        Convert yaml file to dictionary.

        Returns
        -------
        dict
            Returns the converted dictionary with the stored parameters.
        """
        with open(self.filepath, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def store_content(self, store_all: bool = False, skip: list = None) -> dict:
        """
        Store the content of the config file in the database.

        Parameters
        ----------
        store_all : bool
            Define if you want to store all the parameters or only the main ones.
        skip : list
            List of parameters that do not have to be stored.

        Returns
        -------
        dict
            Returns the converted dictionary with the stored parameters.
        """
        config = convert_to_nodes(self.as_dictionary, convert_all=store_all)
        for key, value in config.items():
            if issubclass(type(value), Data) and key not in skip:
                value.store()
        return config

    @property
    def filepath(self) -> str:
        """
        Return the filepath.

        Returns
        -------
        str
            Path of the config file.
        """
        return self.base.attributes.get("filepath")

    @property
    def as_dictionary(self) -> dict:
        """
        Return the filepath.

        Returns
        -------
        dict
            Config file as a dictionary.
        """
        return self.read_yaml()
