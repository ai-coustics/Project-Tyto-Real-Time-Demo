"""Tests for the minimal .env loader."""

import os

from tyto_voice.env import load_env


def test_loads_keys_and_skips_comments(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\n\nAIC_SDK_LICENSE=abc123\nOPENAI_API_KEY='sk-quoted'\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIC_SDK_LICENSE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    path = load_env()
    assert path is not None
    assert os.environ["AIC_SDK_LICENSE"] == "abc123"
    assert os.environ["OPENAI_API_KEY"] == "sk-quoted"  # quotes stripped


def test_existing_env_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("AIC_SDK_LICENSE=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIC_SDK_LICENSE", "from_shell")

    load_env()
    assert os.environ["AIC_SDK_LICENSE"] == "from_shell"


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_env("definitely_missing.env") is None
