"""Test for ModelData class."""

from pathlib import Path

from aiida_mlip.data.model import ModelData

model_path = Path(__file__).parent / "input_files" / "model_local_file.txt"


def test_local_file():
    """Testing that the from_local function works."""
    # Construct a ModelData instance with the local file
    absolute_path = model_path.resolve()
    model = ModelData.from_local(file=absolute_path, architecture="mace")
    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == model_path.read_text(encoding="utf-8")


def test_relativepath():
    """Testing that the from_local function works with a relative path."""
    # Construct a ModelData instance with the local file
    relative_path = model_path.relative_to(Path.cwd())
    model = ModelData.from_local(file=relative_path, architecture="mace")
    # Assert the ModelData contains the content we expect
    content = model.get_content()
    assert content == relative_path.read_text(encoding="utf-8")


def test_architecture():
    """Testing that the architecture is read and added to attributes."""
    model = ModelData.from_local(
        file=model_path,
        filename="model",
        architecture="mace",
    )
    assert model.architecture == "mace"


def test_download_fresh_file_keep(tmp_path):
    """Test if download works and the downloaded file is kept in the chosen folder."""
    # Ensure we do not have the file cached already
    path_test = tmp_path / "mace" / "mace.model"
    path_test.unlink(missing_ok=True)

    # Construct a ModelData instance downloading a non-cached file
    # pylint:disable=line-too-long
    model = ModelData.from_uri(
        uri="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        filename="mace.model",
        cache_dir=tmp_path,
        architecture="mace",
        keep_file=True,
    )

    # Assert the ModelData is downloaded
    assert model.architecture == "mace"
    assert path_test.exists(), f"File {path_test} does not exist."


def test_download_fresh_file(tmp_path):
    """Test if download works and the file is only saved in the database not locally."""
    # Ensure we do not have the file cached already
    path_test = tmp_path / "mace" / "mace.model"
    path_test.unlink(missing_ok=True)

    # Construct a ModelData instance downloading a non-cached file
    # pylint:disable=line-too-long
    model = ModelData.from_uri(
        uri="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        filename="mace.model",
        cache_dir=tmp_path,
        architecture="mace",
    )

    # Assert the ModelData is downloaded
    assert model.architecture == "mace"
    assert path_test.exists() is False, f"File {path_test} exists and shouldn't."


def test_no_download_cached_file(tmp_path):
    """Test if the caching prevents saving duplicate model in the database."""
    # pylint:disable=line-too-long
    existing_model = ModelData.from_uri(
        uri="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        filename="mace_existing.model",
        cache_dir=tmp_path,
        architecture="mace_mp",
    )
    # Construct a ModelData instance that should use the cached file
    # pylint:disable=line-too-long
    model = ModelData.from_uri(
        uri="https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
        cache_dir=tmp_path,
        filename="test_model.model",
        architecture="mace_mp",
    )
    file_path = tmp_path / "test_model.model"

    # Assert the new ModelData was not downloaded and the old one is still there
    assert model.pk == existing_model.pk
    assert model.model_hash == existing_model.model_hash
    assert file_path.exists() is False, f"File {file_path} exists and shouldn't."
