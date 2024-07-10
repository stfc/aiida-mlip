"""Define Model Data type in AiiDA."""

import hashlib
from pathlib import Path
from typing import Any, Optional, Union
from urllib import request

from aiida.orm import QueryBuilder, SinglefileData, load_node
from aiida.tools import delete_nodes


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

    @classmethod
    def _check_existing_file(
        cls, file: Union[str, Path]
    ) -> Path:  # just don't do this, do the hash and then the querybuilder
        """
        Check if a file already exists and return the path of the existing file.

        Parameters
        ----------
        file : Union[str, Path]
            Path to the downloaded model file.

        Returns
        -------
        Path
            The path of the model file of interest (same as input path if no duplicates
            were found).
        """
        file_hash = cls._calculate_hash(file)

        def is_diff_file(curr_path: Path) -> bool:
            """
            Filter to check if two files are different.

            Parameters
            ----------
            curr_path : Path
                Path to the file to compare with.

            Returns
            -------
            bool
                True if the files are different, False otherwise.
            """
            return curr_path.is_file() and not curr_path.samefile(file)

        file_folder = Path(file).parent
        for existing_file in filter(is_diff_file, file_folder.rglob("*")):
            if cls._calculate_hash(existing_file) == file_hash:
                file.unlink()
                return existing_file
        return Path(file)

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
        self.base.attributes.set("filepath", str(file))  # no need

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

    @classmethod  # if I change I won't need this
    def local_file(
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
    def download(  # change names with from (from_local and from_url)
        cls,
        url: str,
        architecture: str,
        filename: Optional[str] = "tmp_file.model",
        cache_dir: Optional[Union[str, Path]] = None,
        keep_file: Optional[bool] = False,
    ):
        """
        Download a file from a URL and save it as ModelData.

        Parameters
        ----------
        url : str
            URL of the file to download.
        architecture : [str]
            Architecture of the mlip model.
        filename : Optional[str], optional
            Name to be used for the file defaults to tmp_file.model.
        cache_dir : Optional[Union[str, Path]], optional
            Path to the folder where the file has to be saved
            (defaults to "~/.cache/mlips/").
        keep_file : Optional[bool], optional
            True to keep the downloaded model, even if there are duplicates)
            (default: False, the file is cancelled and only saved in database).

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
        request.urlretrieve(url, file)

        model = cls.local_file(file=file, architecture=architecture)

        if keep_file:
            return model

        file.unlink(missing_ok=True)

        qb = QueryBuilder()
        qb.append(ModelData, project=["attributes", "pk", "ctime"])

        for i in qb.iterdict():
            if i["ModelData_1"]["attributes"]["model_hash"] == model.model_hash:
                if i["ModelData_1"]["ctime"] != model.ctime:
                    delete_nodes(
                        [model.uuid],
                        dry_run=False,
                        create_forward=True,
                        call_calc_forward=True,
                        call_work_forward=True,
                    )
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
        Return the architecture.

        Returns
        -------
        str
            Architecture of the mlip model.
        """
        return self.base.attributes.get("model_hash")
