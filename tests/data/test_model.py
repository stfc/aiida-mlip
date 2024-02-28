"""Test for ModelData class"""

from pathlib import Path

from aiida_mlip.data.model import ModelData


def test_local_file():
    """Testing that the local file function works"""
    # Construct a ModelData instance with the local file
    model_path = Path(__file__).parent / "input_files" / "model_local_file.txt"
    model = ModelData.local_file(file=model_path, architecture="mace")
    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == model_path.read_text(encoding="utf-8")


def test_architecture():
    """Testing that the architecture is read and added to attributes"""
    file = Path(__file__).parent / "input_files/model_local_file.txt"
    model = ModelData.local_file(
        file=file,
        filename="model",
        architecture="mace",
    )
    assert model.architecture == "mace"


def test_download_fresh_file(tmp_path):
    """Test if download works"""
    # Ensure we do not have the file cached already
    path_test = tmp_path / "mace" / "test_download.txt"
    path_test.unlink(missing_ok=True)

    # Construct a ModelData instance downloading a non-cached file
    model = ModelData.download(
        url="https://raw.githubusercontent.com/stfc/aiida-mlip/main/tests/input_files/file2.txt",
        filename="test_download.txt",
        cache_dir=tmp_path,
        architecture="mace",
    )

    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == "file with content\ncontent2\n"
    file_path = tmp_path / "mace" / "test_download.txt"
    assert file_path.exists(), f"File {file_path} does not exist."


def test_no_download_cached_file(tmp_path):
    """Test if the caching work for avoiding double download"""
    # Ensure file is not already downloaded
    cached_file_path = tmp_path / "test_model.txt"
    cached_file_path.unlink(missing_ok=True)

    # Ensure we have the file cached already
    testdir = tmp_path / "mace"
    testdir.mkdir(parents=True, exist_ok=False)
    (testdir / "test.txt").write_text("file with content\ncontent2\n", encoding="utf-8")

    # Construct a ModelData instance that should use the cached file
    model = ModelData.download(
        url="https://raw.githubusercontent.com/stfc/aiida-mlip/main/tests/input_files/file2.txt",
        cache_dir=tmp_path,
        filename="test_model.txt",
        architecture="mace",
    )

    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == "file with content\ncontent2\n"
    file_path = tmp_path / "mace" / "test_model.txt"
    assert file_path.exists() is False, f"File {file_path} exists but it shouldn't."
    file_path2 = tmp_path / "test_model.txt"
    assert file_path2.exists() is False, f"File {file_path2} exists but it shouldn't."
    file_path3 = tmp_path / "mace" / "test.txt"
    assert file_path3.exists(), f"File {file_path3} should exist"
    assert model.architecture == "mace"
