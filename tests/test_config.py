"""Settings must describe the same layout no matter where the process was started."""

from pathlib import Path

from drdoom.config import PROJECT_ROOT, Settings, get_settings


def test_project_root_is_the_repository_root() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_paths_do_not_depend_on_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    before = Settings().data_dir

    monkeypatch.chdir(tmp_path)
    after = Settings().data_dir

    assert before == after
    assert after.is_absolute()


def test_derived_paths_sit_under_the_project_root() -> None:
    settings = Settings()

    for directory in (*settings.writable_dirs(), settings.data_dir):
        assert directory.is_relative_to(settings.project_root)


def test_environment_overrides_are_read_with_the_expected_prefix(monkeypatch) -> None:
    monkeypatch.setenv("DRDOOM_LOG_LEVEL", "DEBUG")

    assert Settings().log_level == "DEBUG"


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()
