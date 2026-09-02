"""Application settings and filesystem layout.

Every path is derived from the installed package location rather than the process
working directory, so the application behaves identically whether it is started
from the repository root, from a subdirectory, or from inside a container.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


class Settings(BaseSettings):
    """Runtime configuration, overridable through environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="DRDOOM_",
        extra="ignore",
    )

    environment: Literal["local", "ci", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    llm_provider: Literal["groq", "anthropic", "stub"] = "groq"

    @computed_field
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @computed_field
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @computed_field
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field
    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @computed_field
    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    def writable_dirs(self) -> tuple[Path, ...]:
        """Directories the application creates output in."""
        return (self.raw_data_dir, self.processed_data_dir, self.models_dir, self.reports_dir)

    def ensure_directories(self) -> None:
        for directory in self.writable_dirs():
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
