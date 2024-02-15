"""Module docstring TODO"""

from pathlib import Path
from urllib import request
from urllib.parse import urlparse

from aiida.orm import SinglefileData


class ModelData(SinglefileData):
    """Class to handle models"""

    @staticmethod
    def local_file(file: str, filename: str = None):
        """TODO"""
        file_path = Path(file).resolve()
        return ModelData(file=str(file_path), filename=filename)

    @staticmethod
    def download(url: str, filename: str = None, cache_dir: str = None):
        """TODO"""
        if not cache_dir:
            cache_dir = "~/.cache/mace"
        cache_path = Path(cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        model_name = urlparse(url).path.split("/")[-1]
        if filename:
            file = cache_path.joinpath(filename)
        else:
            file = cache_path.joinpath(model_name)
        if not file.exists():
            request.urlretrieve(url, file)

        return ModelData.local_file(file=str(file), filename=filename)
