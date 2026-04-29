"""Tests for loop.config — AgentConfig."""

from the_agents_playbook.loop.config import AgentConfig


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.max_tool_iterations == 25
        assert config.on_error == "abort"
        assert config.entropy_threshold == 1.5
        assert config.max_chain_length == 3

    def test_custom_values(self):
        config = AgentConfig(
            max_tool_iterations=10,
            on_error="yield_and_continue",
            entropy_threshold=2.0,
            max_chain_length=5,
        )
        assert config.max_tool_iterations == 10
        assert config.on_error == "yield_and_continue"
        assert config.entropy_threshold == 2.0
        assert config.max_chain_length == 5

    def test_all_on_error_modes(self):
        for mode in ("raise", "yield_and_continue", "abort"):
            config = AgentConfig(on_error=mode)
            assert config.on_error == mode
