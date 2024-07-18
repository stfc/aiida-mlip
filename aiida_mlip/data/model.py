"""Define Model Data type in AiiDA."""

import hashlib
from pathlib import Path
from typing import Any, Optional, Union
from urllib import request

from aiida.orm import QueryBuilder, SinglefileData, load_node


class ModelData(SinglefileData):
    """
    Define Model Data type in AiiDA.

    Parameters
    ----------
    file : Union[str, Path]
        Absolute path to the file.
    architecture : str
        Architecture of the mlip model.
    filename : Optional[str], optional
        Name to be used for the file (defaults to the name of provided file).

    Attributes
    ----------
    architecture : str
        Architecture of the mlip model.
    model_hash : str
        Hash of the model.

    Methods
    -------
    set_file(file, filename=None, architecture=None, **kwargs)
        Set the file for the node.
    from_local(file, architecture, filename=None):
        Create a ModelData instance from a local file.
    from_uri(uri, architecture, filename=None, cache_dir=None, keep_file=False)
        Download a file from a uri and save it as ModelData.

    Other Parameters
    ----------------
    **kwargs : Any
        Additional keyword arguments.
    """

    @staticmethod
    def _calculate_hash(file: Union[str, Path]) -> str:
        """
        Calculate the hash of a file.

        Parameters
        ----------
        file : Union[str, Path]
            Path to the file for which hash needs to be calculated.

        Returns
        -------
        str
            The SHA-256 hash of the file.
        """
        # Calculate hash
        buf_size = 65536  # reading 64kB (arbitrary) at a time
        sha256 = hashlib.sha256()
        with open(file, "rb") as f:
            # calculating sha in chunks rather than 1 large pass
            while data := f.read(buf_size):
                sha256.update(data)
        file_hash = sha256.hexdigest()
        return file_hash

    def __init__(
        self,
        file: Union[str, Path],
        architecture: str,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ModelData object.

        Parameters
        ----------
        file : Union[str, Path]
            Absolute path to the file.
        architecture : [str]
            Architecture of the mlip model.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).

        Other Parameters
        ----------------
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(file, filename, **kwargs)
        self.base.attributes.set("architecture", architecture)

    def set_file(
        self,
        file: Union[str, Path],
        filename: Optional[str] = None,
        architecture: Optional[str] = None,
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
        architecture : Optional[str], optional
            Architecture of the mlip model.

        Other Parameters
        ----------------
        **kwargs : Any
            Additional keyword arguments.
        """
        super().set_file(file, filename, **kwargs)
        self.base.attributes.set("architecture", architecture)
        # here compute hash and set attribute
        model_hash = self._calculate_hash(file)
        self.base.attributes.set("model_hash", model_hash)

    @classmethod
    def from_local(
        cls,
        file: Union[str, Path],
        architecture: str,
        filename: Optional[str] = None,
    ):
        """
        Create a ModelData instance from a local file.

        Parameters
        ----------
        file : Union[str, Path]
            Path to the file.
        architecture : [str]
            Architecture of the mlip model.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).

        Returns
        -------
        ModelData
            A ModelData instance.
        """
        file_path = Path(file).resolve()
        return cls(file=file_path, architecture=architecture, filename=filename)

    @classmethod
    # pylint: disable=too-many-arguments
    def from_uri(
        cls,
        uri: str,
        architecture: str,
        filename: Optional[str] = "tmp_file.model",
        cache_dir: Optional[Union[str, Path]] = None,
        keep_file: Optional[bool] = False,
    ):
        """
        Download a file from a uri and save it as ModelData.

        Parameters
        ----------
        uri : str
            uri of the file to download.
        architecture : [str]
            Architecture of the mlip model.
        filename : Optional[str], optional
            Name to be used for the file defaults to tmp_file.model.
        cache_dir : Optional[Union[str, Path]], optional
            Path to the folder where the file has to be saved
            (defaults to "~/.cache/mlips/").
        keep_file : Optional[bool], optional
            True to keep the downloaded model, even if there are duplicates.
            (default: False, the file is deleted and only saved in the database).

        Returns
        -------
        ModelData
            A ModelData instance.
        """
        cache_dir = (
            Path(cache_dir) if cache_dir else Path("~/.cache/mlips/").expanduser()
        )
        arch_dir = (cache_dir / architecture) if architecture else cache_dir

        arch_path = arch_dir.resolve()
        arch_path.mkdir(parents=True, exist_ok=True)

        file = arch_path / filename

        # Download file
        request.urlretrieve(uri, file)

        model = cls.from_local(file=file, architecture=architecture)

        if keep_file:
            return model

        file.unlink(missing_ok=True)

        qb = QueryBuilder()
        qb.append(ModelData, project=["attributes", "pk", "ctime"])

        # Looking for ModelData in the whole database
        for i in qb.iterdict():
            # If the hash is the same as the new model, but not the creation time
            # it means that there is already a model that is the same, use that
            if (
                "model_hash" in i["ModelData_1"]["attributes"]
                and i["ModelData_1"]["attributes"]["model_hash"] == model.model_hash
                and i["ModelData_1"]["attributes"]["architecture"] == model.architecture
            ):
                if i["ModelData_1"]["ctime"] != model.ctime:
                    # delete_nodes(
                    #     [model.pk],
                    #     dry_run=False,
                    #     create_forward=True,
                    #     call_calc_forward=True,
                    #     call_work_forward=True,
                    # )
                    model = load_node(i["ModelData_1"]["pk"])
                    break
        return model

    @property
    def architecture(self) -> str:
        """
        Return the architecture.

        Returns
        -------
        str
            Architecture of the mlip model.
        """
        return self.base.attributes.get("architecture")

    @property
    def model_hash(self) -> str:
        """
        Return hash of the architecture.

        Returns
        -------
        str
            Hash of the MLIP model.
        """
        return self.base.attributes.get("model_hash")
