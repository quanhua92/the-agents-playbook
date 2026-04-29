"""Tests for validate_config in settings.py."""

from unittest.mock import patch

import pytest

from the_agents_playbook.settings import Settings, validate_config


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        openai_api_key="",
        openai_base_url="https://openrouter.ai/api/v1",
        openai_model="openai/gpt-oss-20b",
        anthropic_api_key="",
        anthropic_base_url="https://api.anthropic.com/v1",
        anthropic_model="claude-sonnet-4-6",
        embedding_api_key="",
        embedding_base_url="https://openrouter.ai/api/v1",
        embedding_model="openai/text-embedding-3-small",
        mock_only=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestValidateConfig:
    def test_clean_config_no_warnings(self):
        cfg = _make_settings(
            openai_api_key="sk-or-v1-some-key",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert warnings == []

    def test_malformed_openai_key(self):
        cfg = _make_settings(
            openai_api_key="not-sk-prefix",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert any("OPENAI_API_KEY looks malformed" in w for w in warnings)

    def test_valid_openai_key_no_warning(self):
        cfg = _make_settings(
            openai_api_key="sk-or-v1-abcdef",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert not any("OPENAI_API_KEY looks malformed" in w for w in warnings)

    def test_malformed_anthropic_key(self):
        cfg = _make_settings(
            anthropic_api_key="not-sk-ant-prefix",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert any("ANTHROPIC_API_KEY looks malformed" in w for w in warnings)

    def test_valid_anthropic_key_no_warning(self):
        cfg = _make_settings(
            anthropic_api_key="sk-ant-api03-abcdef",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert not any("ANTHROPIC_API_KEY looks malformed" in w for w in warnings)

    def test_key_url_mismatch_openai_key_anthropic_url(self):
        cfg = _make_settings(
            openai_api_key="sk-or-v1-key",
            openai_base_url="https://api.anthropic.com/v1",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert any("anthropic" in w.lower() and "openai" in w.lower() for w in warnings)

    def test_key_url_mismatch_anthropic_key_openai_url(self):
        cfg = _make_settings(
            anthropic_api_key="sk-ant-key",
            anthropic_base_url="https://api.openai.com/v1",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert any("anthropic" in w.lower() and "openai" in w.lower() for w in warnings)

    def test_no_embedding_key_warning(self):
        cfg = _make_settings(
            openai_api_key="sk-or-v1-key",
            embedding_api_key="",
            mock_only=False,
        )
        warnings = validate_config(cfg)
        assert any("EMBEDDING_API_KEY" in w for w in warnings)

    def test_embedding_key_set_no_warning(self):
        cfg = _make_settings(
            openai_api_key="sk-or-v1-key",
            embedding_api_key="sk-or-v1-key",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert not any("EMBEDDING_API_KEY" in w for w in warnings)

    def test_anthropic_model_on_openrouter_url(self):
        cfg = _make_settings(
            openai_model="anthropic/claude-sonnet-4-6",
            openai_base_url="https://openrouter.ai/api/v1",
            openai_api_key="sk-or-v1-key",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        # This should NOT warn because openrouter does route anthropic models
        assert not any("anthropic" in w.lower() and "may not route" in w.lower() for w in warnings)

    def test_anthropic_model_on_non_anthropic_url(self):
        cfg = _make_settings(
            openai_model="anthropic/claude-sonnet-4-6",
            openai_base_url="https://api.someotherprovider.com/v1",
            openai_api_key="sk-or-v1-key",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert any("anthropic/" in w for w in warnings)

    def test_uses_global_settings_when_no_arg(self):
        """When called with no args, validate_config uses the global settings."""
        # Just verify it doesn't crash — we can't easily control global settings here
        # since it's loaded from .env. Just verify it returns a list.
        result = validate_config()
        assert isinstance(result, list)

    def test_multiple_warnings(self):
        cfg = _make_settings(
            openai_api_key="bad-key",
            anthropic_api_key="bad-key",
            openai_base_url="https://api.anthropic.com/v1",
            mock_only=True,
        )
        warnings = validate_config(cfg)
        assert len(warnings) >= 2
