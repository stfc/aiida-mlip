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
        Path of the mlip model.

    Methods
    -------
    set_file(file, filename=None, architecture=None, **kwargs)
        Set the file for the node.
    local_file(file, architecture, filename=None):
        Create a ModelData instance from a local file.
    download(url, architecture, filename=None, cache_dir=None, force_download=False)
        Download a file from a URL and save it as ModelData.

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
            Path of the mlip model.
        """
        return self.base.attributes.get("filepath")

    @property
    def as_dictionary(self) -> dict:
        """
        Return the filepath.

        Returns
        -------
        str
            Path of the mlip model.
        """
        return dict(self.read_yaml())
