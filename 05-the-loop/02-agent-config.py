"""02-agent-config.py — Configure AgentConfig (max iterations, error policy).

AgentConfig controls the ReAct loop behavior: how many tool iterations
are allowed, what happens on provider errors, and entropy thresholds
for routing decisions.
"""

from the_agents_playbook.loop import AgentConfig


def main():
    # --- Default configuration ---

    config = AgentConfig()
    print("Default AgentConfig:")
    print(f"  max_tool_iterations: {config.max_tool_iterations}")
    print(f"  on_error:            {config.on_error!r}")
    print(f"  entropy_threshold:   {config.entropy_threshold}")
    print(f"  max_chain_length:    {config.max_chain_length}")
    print()

    # --- Conservative configuration (safer, slower) ---

    safe_config = AgentConfig(
        max_tool_iterations=5,
        on_error="abort",
        entropy_threshold=1.0,
        max_chain_length=2,
    )
    print("Safe AgentConfig:")
    print(f"  max_tool_iterations: {safe_config.max_tool_iterations}")
    print(f"  on_error:            {safe_config.on_error!r}")
    print(f"  entropy_threshold:   {safe_config.entropy_threshold}")
    print(f"  max_chain_length:    {safe_config.max_chain_length}")
    print()

    # --- Aggressive configuration (more autonomous) ---

    fast_config = AgentConfig(
        max_tool_iterations=50,
        on_error="yield_and_continue",
        entropy_threshold=2.5,
        max_chain_length=5,
    )
    print("Fast AgentConfig:")
    print(f"  max_tool_iterations: {fast_config.max_tool_iterations}")
    print(f"  on_error:            {fast_config.on_error!r}")
    print(f"  entropy_threshold:   {fast_config.entropy_threshold}")
    print(f"  max_chain_length:    {fast_config.max_chain_length}")
    print()

    # --- All error policies ---

    print("Available on_error modes:")
    for mode in ("raise", "yield_and_continue", "abort"):
        cfg = AgentConfig(on_error=mode)
        print(f"  {mode:20s} → {cfg.on_error}")


if __name__ == "__main__":
    main()
