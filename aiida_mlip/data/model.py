"""Define Model Data type in AiiDA"""

import hashlib
from pathlib import Path
from urllib import request
from urllib.parse import urlparse

from aiida.orm import SinglefileData


def _calculate_hash(file: str | Path) -> str:
    """Function to calculate file hashes
    param: file: path to the file you want to calculate the hash for (str)
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
    arch_path: str | Path, file: str | Path, architecture: str, cache_path: str | Path
) -> tuple[Path, str]:
    """Function to check if a file already exists and return a file path and architecture
    param arch_path: Path to the folder in which the file is stored (str or pathlib.Path).
    param file: Path to the file (str or pathlib.Path)
    param architecture: mlip architecture of the model
    param cache_path: Path to the parent folder of arch_path (str or pathlib.Path)
    return: A tuple of a PAth and a str containing the path of the file of interest and the architecture
    """
    file_hash = _calculate_hash(file)
    for existing_file in (
        ex_file
        for ex_file in arch_path.rglob("*")
        if ex_file.is_file() and not ex_file.samefile(file)
    ):
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
        self, file: str | Path, filename: str = None, architecture: str = None, **kwargs
    ):
        """Construct a new instance and set the contents to that of the file.

        :param file: absolute path to the file
        :param filename: specify filename to use (defaults to name of provided file).
        :param architecture: specify architecture
        """

        super().__init__(file, filename, **kwargs)
        self.base.attributes.set("architecture", architecture)

    def set_file(
        self, file: str | Path, filename: str = None, architecture: str = None, **kwargs
    ):
        """Add a file to the node, parse it and set the attributes found.

        :param file: absolute path to the file (str or pathlib.Path)
        :param filename: specify filename to use (defaults to name of provided file)
        :param architecture: specify architecture in which the model is used (default = None)
        """

        super().set_file(file, filename, **kwargs)

        self.base.attributes.set("architecture", architecture)

    @classmethod
    def local_file(
        cls, file: str | Path, filename: str = None, architecture: str = None
    ):
        """Save a local file as ModelData
        :param file: path to the file (str or pathlib.Path)
        :param filename: specify filename to use (defaults to name of provided file)
        :param architecture: specify architecture in which the model is used (default = None)
        """
        file_path = Path(file).resolve()
        return cls(file=file_path, filename=filename, architecture=architecture)

    @classmethod
    # pylint: disable=R0913
    def download(
        cls,
        url: str,
        filename: str = None,
        cache_dir: str | Path = None,
        architecture: str = None,
        force_download: bool = False,
    ):
        """Download a file from a URL and save it as a ModelData
        :param url: url of the file to download
        :param filename: specify filename to use (defaults to name of provided file)
        :param cache_dir: path (str or pathlib.Path) to the folder where the file has to be saved (default "~/.cache/mlips/")
        :param architecture: specify architecture in which the model is used (default = None)
        :param force_download: True to keep the downloaded model even if there are duplicates (default: False)

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
            arch_path, file, architecture, cache_path
        )

        return cls.local_file(file=filepath, architecture=architecture)

    @property
    def architecture(self) -> str:
        """Return architecture

        :return: a string
        """
        return self.base.attributes.get("architecture")
