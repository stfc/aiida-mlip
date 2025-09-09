"""Define Model Data type in AiiDA."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib import request

from aiida.orm import QueryBuilder, SinglefileData, load_node


class ModelData(SinglefileData):
    """
    Define Model Data type in AiiDA.

    Parameters
    ----------
    file : str | Path
        Absolute path to the file.
    architecture : str
        Architecture of the mlip model.
    filename : str | None
        Name to be used for the file. Default is the name of provided file.

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
        Download a file from a URI and save it as ModelData.

    Other Parameters
    ----------------
    **kwargs : Any
        Additional keyword arguments.
    """

    @staticmethod
    def _calculate_hash(file: str | Path) -> str:
        """
        Calculate the hash of a file.

        Parameters
        ----------
        file : str | Path
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
        return sha256.hexdigest()

    def __init__(
        self,
        file: str | Path,
        architecture: str,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ModelData object.

        Parameters
        ----------
        file : str | Path
            Absolute path to the file.
        architecture : str
            Architecture of the mlip model.
        filename : str | None
            Name to be used for the file. Default is the name of provided file.

        Other Parameters
        ----------------
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(file, filename, **kwargs)
        self.base.attributes.set("architecture", architecture)

    def set_file(
        self,
        file: str | Path,
        filename: str | None = None,
        architecture: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Set the file for the node.

        Parameters
        ----------
        file : str | Path
            Absolute path to the file.
        filename : str | None
            Name to be used for the file. Defaults is the name of provided file.
        architecture : str | None
            Architecture of the mlip model. Default is `None`.

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
        file: str | Path,
        architecture: str,
        filename: str | None = None,
    ):
        """
        Create a ModelData instance from a local file.

        Parameters
        ----------
        file : str | Path
            Path to the file.
        architecture : str
            Architecture of the mlip model.
        filename : str | None
            Name to be used for the file. Default is the name of provided file.

        Returns
        -------
        ModelData
            A ModelData instance.
        """
        file_path = Path(file).resolve()
        return cls(file=file_path, architecture=architecture, filename=filename)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        architecture: str,
        filename: str | None = "tmp_file.model",
        cache_dir: str | Path | None = None,
        keep_file: bool | None = False,
    ):
        """
        Download a file from a URI and save it as ModelData.

        Parameters
        ----------
        uri : str
            URI of the file to download.
        architecture : str
            Architecture of the mlip model.
        filename : str | None
            Name to be used for the file defaults to tmp_file.model.
        cache_dir : str | Path | None
            Path to the folder where the file has to be saved. Defaults is
            "~/.cache/mlips/".
        keep_file : bool | None
            True to keep the downloaded model, even if there are duplicates.
            Default is `False`, so the file is deleted and only saved in the database.

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

        # Check if the same model was used previously
        qb = QueryBuilder()
        qb.append(
            ModelData,
            filters={
                "attributes.model_hash": model.model_hash,
                "attributes.architecture": model.architecture,
                "ctime": {"!in": [model.ctime]},
            },
            project=["attributes", "pk", "ctime"],
        )

        if qb.count() != 0:
            model = load_node(
                qb.first()[1]
            )  # This gets the pk of the first model in the query

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
