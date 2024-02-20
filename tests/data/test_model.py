"""Test for ModelData class"""

from pathlib import Path
import shutil

from aiida_mlip.data.model import ModelData


def test_local_file():
    """Testing that the local file function works"""
    # Construct a ModelData instance with the local file
    model = ModelData.local_file("./tests/input_files/model_local_file.txt")
    # Assert the ModelData contains the content we expect
    content = model.get_content()
    with open("./tests/input_files/model_local_file.txt", encoding="utf-8") as f:
        assert content == f.read()


def test_architecture():
    """Testing that the architecture is read and added to attributes"""
    model = ModelData.local_file(
        "./tests/input_files/model_local_file.txt",
        filename="model",
        architecture="mace",
    )
    assert model.architecture == "mace"


def test_download_fresh_file():
    """Test if download works"""
    # Ensure we do not have the file cached already
    path_test = Path("./tests/data/tmp/mace/test_download.txt")
    if path_test.exists():
        path_test.unlink()

    # Construct a ModelData instance downloading a non-cached file
    model = ModelData.download(
        url="https://raw.githubusercontent.com/stfc/aiida-mlip/main/tests/input_files/file2.txt",
        filename="test_download.txt",
        cache_dir="./tests/data/tmp",
        architecture="mace",
    )

    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == "file with content\ncontent2\n"
    file_path = Path("./tests/data/tmp/mace/test_download.txt")
    assert file_path.exists(), f"File {file_path} does not exists."
    # Clear test cache
    shutil.rmtree(Path("./tests/data/tmp/mace/"))


def test_no_download_cached_file():
    """Test if the caching work for avoiding double download"""
    # Ensure file is not already downloaded
    cached_file_path = Path("./tests/data/tmp/test_modell.txt")
    if cached_file_path.exists():
        cached_file_path.unlink()

    # Ensure we have the file cached already
    testdir = Path("./tests/data/tmp/mace/")
    testdir.mkdir(parents=True, exist_ok=False)
    with open("./tests/data/tmp/mace/test.txt", "w", encoding="utf-8") as f:
        f.write("file with content\ncontent2\n")

    # Construct a ModelData instance that should use the cached file
    model = ModelData.download(
        url="https://raw.githubusercontent.com/stfc/aiida-mlip/main/tests/input_files/file2.txt",
        cache_dir="./tests/data/tmp/",
        filename="test_modell.txt",
        # architecture = "mace"
    )

    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == "file with content\ncontent2\n"
    file_path = Path("./tests/data/tmp/mace/test_modell.txt")
    assert file_path.exists() is False, f"File {file_path} exists but it shouldn't."
    file_path2 = Path("./tests/data/tmp/test_modell.txt")
    assert file_path2.exists() is False, f"File {file_path2} exists but it shouldn't."
    file_path3 = Path("./tests/data/tmp/mace/test.txt")
    assert file_path3.exists() is True, f"File {file_path3} should exists"
    assert model.architecture == "mace"

    # Clear test cache
    shutil.rmtree(Path("./tests/data/tmp/"))
