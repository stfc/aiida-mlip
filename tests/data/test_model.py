"""Test for ModelData class"""

from pathlib import Path

from aiida_mlip.data.model import ModelData


def test_local_file():
    """Testing that the local file function works"""
    # Construct a ModelData instance with the local file
    model_path = Path(__file__).parent / "input_files" / "model_local_file.txt"
    absolute_path = model_path.resolve()
    model = ModelData.local_file(file=absolute_path, architecture="mace")
    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == model_path.read_text(encoding="utf-8")


def test_relativepath():
    """Testing that the local file function works"""
    # Construct a ModelData instance with the local file
    model_path = Path(__file__).parent / "input_files" / "model_local_file.txt"
    relative_path = model_path.relative_to(Path.cwd())
    model = ModelData.local_file(file=relative_path, architecture="mace")
    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == relative_path.read_text(encoding="utf-8")


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
    path_test = tmp_path / "mace" / "mace.model"
    path_test.unlink(missing_ok=True)

    # Construct a ModelData instance downloading a non-cached file
    # pylint:disable=line-too-long
    model = ModelData.download(
        url="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        filename="mace.model",
        cache_dir=tmp_path,
        architecture="mace",
    )

    # Assert the ModelData is downloaded
    file_path = tmp_path / "mace" / "mace.model"
    assert model.architecture == "mace"
    assert file_path.exists(), f"File {file_path} does not exist."


def test_no_download_cached_file():
    """Test if the caching work for avoiding double download"""

    # Construct a ModelData instance that should use the cached file
    # pylint:disable=line-too-long
    model = ModelData.download(
        url="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        cache_dir=Path(__file__).parent / "input_files",
        filename="test_model.model",
        architecture="mace",
    )

    # Assert the new ModelData was not downloaded and the old one is still there
    file_path = Path(__file__).parent / "input_files" / "mace" / "test_model.model"
    assert not file_path.exists(), f"File {file_path} exists but it shouldn't."
    file_path2 = Path(__file__).parent / "input_files" / "test_model.model"
    assert not file_path2.exists(), f"File {file_path2} exists but it shouldn't."
    file_path3 = Path(__file__).parent / "input_files" / "mace" / "mace_mp_small.model"
    assert file_path3.exists(), f"File {file_path3} should exist"
    assert model.architecture == "mace"
