"""Offline checks for OpenRouter credentials and request routing."""

from types import SimpleNamespace
import sys

import pytest

from evaloop.core.config import EvaluationConfig


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_parent_dotenv_routes_non_openai_model_to_openrouter(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-router-key\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    config = EvaluationConfig(
        model="qwen/example-model", provider="openrouter", max_new_tokens=512
    )
    request_config = config.get_model_config()
    assert request_config["type"] == "openai"
    assert request_config["model_id"] == "qwen/example-model"
    assert request_config["api_key"] == "test-router-key"
    assert request_config["base_url"] == "https://openrouter.ai/api/v1"
    assert request_config["max_tokens"] == 512
    assert config.to_dict()["provider"] == "openrouter"
    assert "api_key" not in config.to_dict()


def test_explicit_dotenv_does_not_override_exported_key(tmp_path, monkeypatch):
    env_file = tmp_path / "credentials.env"
    env_file.write_text("OPENROUTER_API_KEY=file-key\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "exported-key")
    config = EvaluationConfig(
        model="google/example-model", provider="openrouter", env_file=str(env_file)
    )
    assert config.get_model_config()["api_key"] == "exported-key"


def test_missing_router_key_does_not_fall_back_to_openai_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        EvaluationConfig(
            model="qwen/example-model", provider="openrouter",
            env_file=str(env_file),
        )


def test_missing_explicit_dotenv_fails(tmp_path):
    with pytest.raises(ValueError, match="Environment file not found"):
        EvaluationConfig(model="gpt-example", env_file=str(tmp_path / "missing.env"))


def test_default_openai_routing_is_unchanged(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    config = EvaluationConfig(model="gpt-example").get_model_config()
    assert config["type"] == "openai"
    assert config["api_key"] == "test-openai-key"
    assert "base_url" not in config


def test_router_settings_reach_client_and_request(monkeypatch):
    from evaloop.models import closed_source

    calls = {}

    def complete(**kwargs):
        calls["request"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="generated code"))]
        )

    def client(**kwargs):
        calls["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))

    monkeypatch.setattr(closed_source, "OpenAI", client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example/v1")
    config = EvaluationConfig(
        model="qwen/example-model", provider="openrouter", reasoning_enabled=False
    )
    model = closed_source.OpenAILLM(config.get_model_config())
    assert model.generate("Write a function") == "generated code"
    assert calls["client"] == {
        "api_key": "test-router-key", "base_url": "https://openrouter.ai/api/v1"
    }
    assert calls["request"]["model"] == "qwen/example-model"
    assert calls["request"]["temperature"] == 0
    assert calls["request"]["max_tokens"] == 1024
    assert calls["request"]["extra_body"]["reasoning"] == {"enabled": False}


def test_cli_passes_provider_and_environment_file(tmp_path, monkeypatch):
    from evaloop.cli import EvaLoopCLI

    env_file = tmp_path / "credentials.env"
    env_file.write_text("OPENROUTER_API_KEY=test-router-key\n")
    captured = {}

    class OfflineEvaluator:
        def __init__(self, config):
            captured["config"] = config.get_model_config()
            captured["loop_type"] = config.loop_type

        def run_code_generation_summarization(self):
            return ["offline-result"]

    monkeypatch.setitem(
        sys.modules, "evaloop.core.evaluator",
        SimpleNamespace(EvaLoopEvaluator=OfflineEvaluator),
    )
    result = EvaLoopCLI().evaluate(
        model="google/example-model", provider="openrouter", env_file=str(env_file),
        loop_type="pseudocode",
        api_concurrency=4,
    )
    assert result == ["offline-result"]
    assert captured["config"]["type"] == "openai"
    assert captured["config"]["api_key"] == "test-router-key"
    assert captured["config"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["loop_type"] == "pseudocode"
    assert captured["config"]["api_concurrency"] == 4
