"""Define Model Data type in AiiDA"""

import hashlib
from pathlib import Path
from urllib import request
from urllib.parse import urlparse

from aiida.orm import SinglefileData


def calculate_hash(file):
    """Function to calculate file hashes"""
    # Calculate hash
    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        for data in iter(lambda: f.read(65536), b""):
            sha256.update(data)
    file_hash = sha256.hexdigest()
    return file_hash


def check_existing_file(arch_path, file, file_hash, architecture, cache_path):
    """Function to check if a file already exists and return a file path and architecture"""
    for existing_file in arch_path.rglob("*"):
        if existing_file.is_file() and existing_file != file:
            existing_file_hash = calculate_hash(existing_file)
            if existing_file_hash == file_hash:
                ex_file_path = Path(existing_file).parent
                if arch_path == ex_file_path:
                    file.unlink()
                    return existing_file, architecture

                if arch_path == cache_path and arch_path != ex_file_path:
                    architecture = ex_file_path.name
                    file.unlink()
                    return existing_file, architecture
    return file, architecture


class ModelData(SinglefileData):
    """Class to handle models"""

    def __init__(self, file, filename, architecture: str = None, **kwargs):
        """Add a file to the node, parse it and set the attributes found.

        :param file: absolute path to the file or a filelike object
        :param filename: specify filename to use (defaults to name of provided file).
        :param architecture: specify architecture
        """

        super().__init__(file, filename, **kwargs)
        self.base.attributes.set("architecture", architecture)

    def set_file(self, file, filename=None, architecture: str = None, **kwargs):
        """Add a file to the node, parse it and set the attributes found.

        :param file: absolute path to the file or a filelike object
        :param filename: specify filename to use (defaults to name of provided file).
        """

        super().set_file(file, filename, **kwargs)

        self.base.attributes.set("architecture", architecture)

    @classmethod
    def local_file(cls, file: str, filename: str = None, architecture: str = None):
        """Sve a local file as ModelData"""
        file_path = Path(file).resolve()
        return cls(file=str(file_path), filename=filename, architecture=architecture)

    @classmethod
    def download(
        cls,
        url: str,
        filename: str = None,
        cache_dir: str = None,
        architecture: str = None,
    ):
        """Download a file from a URL and save it as a ModelData"""
        if not cache_dir:
            cache_dir = "~/.cache/mlips/"
        if architecture is None:
            arch_dir = Path(cache_dir)
        else:
            arch_dir = Path(cache_dir).joinpath(architecture)

        cache_path = Path(cache_dir).resolve()
        arch_path = arch_dir.resolve()
        arch_path.mkdir(parents=True, exist_ok=True)

        model_name = urlparse(url).path.split("/")[-1]

        if filename:
            file = arch_path.joinpath(filename)
        else:
            file = arch_path.joinpath(model_name)

        # Check if there is already a file named that way and rename it
        if file.exists():
            file = file.with_name(file.stem + "_2" + file.suffix)
            # Check again if the new filename already exists, just in case
            while file.exists():
                # If the new filename also exists, keep incrementing the number
                file = file.with_name(
                    file.stem[:-2] + str(int(file.stem[-2:]) + 1).zfill(2) + file.suffix
                )
        # Download file
        request.urlretrieve(url, file)

        file_hash = calculate_hash(file)

        # Check if the hash matches any other file in the directory
        f, architecture = check_existing_file(
            arch_path, file, file_hash, architecture, cache_path
        )

        return cls.local_file(file=str(f), architecture=architecture)

    @property
    def architecture(self):
        """Return architecture

        :return: a string
        """
        return self.base.attributes.get("architecture")
