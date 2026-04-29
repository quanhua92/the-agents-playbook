#!/usr/bin/env bash
# Run all LangGraph examples.
# Examples requiring API keys are marked with [API] and will make real LLM calls.
# Other examples use mocks or don't need API access.

set -e
cd "$(dirname "$0")"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

run_example() {
    local file="$1"
    local name="$2"
    local desc="$3"

    printf "  %-50s " "$name"
    if uv run python "$file" > /tmp/lg_example_output.txt 2>&1; then
        echo "OK"
        RESULTS+=("OK  $name")
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        RESULTS+=("FAIL $name")
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================="
echo "  LangGraph Examples Runner"
echo "========================================="
echo ""

# 01-basic-calls
echo "--- 01-basic-calls ---"
run_example "01-basic-calls/01_basic_chat.py"     "01_basic_chat [API]"     "Basic ChatOpenAI/ChatAnthropic call"
run_example "01-basic-calls/02_structured_output.py" "02_structured_output [API]" "Structured output with Pydantic"
run_example "01-basic-calls/03_tool_calling.py"     "03_tool_calling [API]"     "Native tool calling with @tool"
run_example "01-basic-calls/04_streaming.py"         "04_streaming [API]"         "Token-by-token streaming"
echo ""

# 02-tools
echo "--- 02-tools ---"
run_example "02-tools/01_tool_protocol.py"   "01_tool_protocol [API]"    "@tool vs Tool ABC"
run_example "02-tools/02_tool_node.py"       "02_tool_node [API]"        "Tool dispatch (ToolNode)"
run_example "02-tools/03_bound_tools.py"     "03_bound_tools [API]"      "bind_tools for LLM-driven selection"
echo ""

# 03-memory
echo "--- 03-memory ---"
run_example "03-memory/01_checkpoint.py"      "01_checkpoint [API]"       "MemorySaver conversation persistence"
run_example "03-memory/02_thread_history.py"   "02_thread_history [API]"   "Multi-thread conversations"
run_example "03-memory/03_long_context.py"     "03_long_context"           "SessionCompactor (no API)"
echo ""

# 04-state
echo "--- 04-state ---"
run_example "04-state/01_typed_state.py"      "01_typed_state"            "TypedDict state (no API)"
run_example "04-state/02_reducer_state.py"    "02_reducer_state"          "State reducers (no API)"
run_example "04-state/03_context_layers.py"   "03_context_layers"         "Context layer composition (no API)"
echo ""

# 05-the-agent
echo "--- 05-the-agent ---"
run_example "05-the-agent/01_simple_graph.py"      "01_simple_graph"           "Basic StateGraph (no API)"
run_example "05-the-agent/02_react_agent.py"       "02_react_agent [API]"       "create_react_agent"
run_example "05-the-agent/03_conditional_edges.py"  "03_conditional_edges"      "Conditional routing (no API)"
run_example "05-the-agent/04_tool_chaining.py"     "04_tool_chaining [API]"     "Multi-step tool use"
echo ""

# 06-human-in-the-loop
echo "--- 06-human-in-the-loop ---"
run_example "06-human-in-the-loop/01_interrupt.py"        "01_interrupt [API]"         "interrupt() for approval"
run_example "06-human-in-the-loop/02_command_resuming.py" "02_command_resuming [API]"  "Command-based resumption"
echo ""

# 07-workflows
echo "--- 07-workflows ---"
run_example "07-workflows/01_plan_execute.py" "01_plan_execute"         "Plan-and-execute (no API)"
run_example "07-workflows/02_parallel_nodes.py" "02_parallel_nodes"       "Send() parallelism (no API)"
echo ""

# 08-the-claw
echo "--- 08-the-claw ---"
run_example "08-the-claw/01_error_edges.py" "01_error_edges"          "Error handling edges (no API)"
run_example "08-the-claw/02_retry_loop.py"  "02_retry_loop"           "Retry with backoff (no API)"
run_example "08-the-claw/03_evaluation.py"  "03_evaluation [API]"      "Evaluation harness"
echo ""

# Summary
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================="
if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Failed examples:"
    for r in "${RESULTS[@]}"; do
        if [[ "$r" == FAIL* ]]; then
            echo "  $r"
        fi
    done
    exit 1
fi
