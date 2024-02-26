"""Define Model Data type in AiiDA"""

import hashlib
from pathlib import Path
from typing import Any, Optional, Union
from urllib import request
from urllib.parse import urlparse

from aiida.orm import SinglefileData


def _calculate_hash(file: Union[str, Path]) -> str:
    """Calculate the hash of a file.

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


def _check_existing_file(
    arch_path: Union[str, Path],
    file: Union[str, Path],
    cache_path: Union[str, Path],
    architecture: Optional[str] = None,
) -> tuple[Path, str]:
    """Check if a file already exists and return its path and architecture.

    Parameters
    ----------
    arch_path : Union[str, Path]
        Path to the folder containing the file.
    file : Union[str, Path]
        Path to the file.
    cache_path : Union[str, Path]
        Path to the parent folder of arch_path.
    architecture : Optional[str]
        MLIP architecture of the model.

    Returns
    -------
    tuple[Path, str]
        A tuple containing the path of the file of interest and its architecture.
    """
    file_hash = _calculate_hash(file)

    def is_diff_file(curr_path: Path):
        return curr_path.is_file() and not curr_path.samefile(file)

    for existing_file in filter(is_diff_file, arch_path.rglob("*")):
        if _calculate_hash(existing_file) == file_hash:
            ex_file_path = Path(existing_file).parent
            if arch_path == ex_file_path:
                file.unlink()
                return existing_file, architecture

            if arch_path == cache_path and arch_path != ex_file_path:
                architecture = ex_file_path.name
                file.unlink()
                return Path(existing_file), architecture
    return Path(file), architecture


class ModelData(SinglefileData):
    """Class to save a model file as an AiiDA data type.
    The file can be an existing one or a file to download
    """

    def __init__(
        self,
        file: Union[str, Path],
        filename: Optional[str] = None,
        architecture: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ModelData object.

        Parameters
        ----------
        file : Union[str, Path]
            Absolute path to the file.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).
        architecture : Optional[str], optional
            Architecture information.

        Other Parameters
        ----------------
        kwargs : Any
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
        """Set the file for the node.

        Parameters
        ----------
        file : Union[str, Path]
            Absolute path to the file.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).
        architecture : Optional[str], optional
            Architecture.

        Other Parameters
        ----------------
        kwargs : Any
            Additional keyword arguments.
        """

        super().set_file(file, filename, **kwargs)

        self.base.attributes.set("architecture", architecture)

    @classmethod
    def local_file(
        cls,
        file: Union[str, Path],
        filename: Optional[str] = None,
        architecture: Optional[str] = None,
    ):
        """Create a ModelData instance from a local file.

        Parameters
        ----------
        file : Union[str, Path]
            Path to the file.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).
        architecture : Optional[str], optional
            Architecture.

        Returns
        -------
        ModelData
            A ModelData instance.
        """

        file_path = Path(file).resolve()
        return cls(file=file_path, filename=filename, architecture=architecture)

    @classmethod
    # pylint: disable=too-many-arguments
    def download(
        cls,
        url: str,
        filename: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        architecture: Optional[str] = None,
        force_download: Optional[bool] = False,
    ):
        """Download a file from a URL and save it as ModelData.

        Parameters
        ----------
        url : str
            URL of the file to download.
        filename : Optional[str], optional
            Name to be used for the file (defaults to the name of provided file).
        cache_dir : Optional[Union[str, Path]], optional
            Path to the folder where the file has to be saved (defaults to "~/.cache/mlips/").
        architecture : Optional[str], optional
            Architecture.
        force_download : Optional[bool], optional
            True to keep the downloaded model even if there are duplicates (default: False).

        Returns
        -------
        ModelData
            A ModelData instance.
        """

        cache_dir = Path(cache_dir if cache_dir else "~/.cache/mlips/")
        arch_dir = cache_dir / architecture if architecture else cache_dir

        cache_path = cache_dir.resolve()
        arch_path = arch_dir.resolve()
        arch_path.mkdir(parents=True, exist_ok=True)

        model_name = urlparse(url).path.split("/")[-1]

        file = arch_path
        if filename:
            file /= filename
        else:
            file /= model_name

        # Check if there is already a file named that way and rename it
        stem = file.stem
        i = 1
        while file.exists():
            i += 1
            file = file.with_stem(f"{stem}_{i}")

        # Download file
        request.urlretrieve(url, file)

        if force_download:
            print(f"filename changed to {file}")
            return cls.local_file(file=file, architecture=architecture)

        # Check if the hash of the just downloaded file matches any other file in the directory
        filepath, architecture = _check_existing_file(
            arch_path, file, cache_path, architecture
        )

        return cls.local_file(file=filepath, architecture=architecture)

    @property
    def architecture(self) -> str:
        """Return the architecture.

        Returns
        -------
        str
            Architecture.
        """

        return self.base.attributes.get("architecture")
