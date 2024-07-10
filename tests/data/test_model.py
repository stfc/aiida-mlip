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


def test_download_fresh_file_keep(tmp_path):
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
        keep_file=True,
    )

    # Assert the ModelData is downloaded
    file_path = tmp_path / "mace" / "mace.model"
    assert model.architecture == "mace"
    assert file_path.exists(), f"File {file_path} does not exist."


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
    assert file_path.exists() is False, f"File {file_path} exists and shouldn't."


def test_no_download_cached_file(tmp_path):
    """Test if the caching work for avoiding double download"""

    # pylint:disable=line-too-long
    existing_model = ModelData.download(
        url="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        filename="mace_existing.model",
        cache_dir=tmp_path,
        architecture="mace_mp",
    )
    # Construct a ModelData instance that should use the cached file
    # pylint:disable=line-too-long
    model = ModelData.download(
        url="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        cache_dir=tmp_path,
        filename="test_model.model",
        architecture="mace_mp",
    )

    # Assert the new ModelData was not downloaded and the old one is still there
    assert model.pk == existing_model.pk
