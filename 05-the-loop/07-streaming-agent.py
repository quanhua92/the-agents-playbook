"""07-streaming-agent.py — Real-time streaming agent loop.

The standard run() method waits for the full LLM response before yielding
anything. run_streaming() uses provider.stream() to yield text deltas
token-by-token, interleaved with tool call events.

Key concepts:
- Text deltas arrive token-by-token for responsive UX
- Tool call chunks are buffered until complete, then dispatched
- The streaming loop handles interleaved text and tool calls gracefully

This is a conceptual demo showing how run_streaming differs from run.
Without a live provider, we simulate the streaming behavior.
"""


def main():
    print("=== Streaming Agent Loop: Conceptual Demo ===\n")

    # --- Show the event flow difference ---
    print("--- Non-streaming (run) ---")
    print("User: What is 15 * 23?\n")
    print("Events (appear all at once after LLM finishes):")
    print("  [status] Thinking (iteration 1)...")
    print("  [text]   15 * 23 = 345")
    print("  (User waits ~3-5 seconds of silence, then sees everything)\n")

    print("--- Streaming (run_streaming) ---")
    print("User: What is 15 * 23?\n")
    print("Events (arrive in real-time):")
    print("  [status]     Thinking (iteration 1)...")

    # Simulate streaming text deltas
    import time
    response = "15 multiplied by 23 equals 345."
    for char in response:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print("\n  (User sees each token as it arrives — no silence)\n")

    # --- Show interleaved text + tool call streaming ---
    print("--- Interleaved Text and Tool Calls ---\n")
    print("User: What is the weather in Tokyo? Then calculate 100 / 7.\n")

    # Simulate: text, then tool call, then more text
    print("  [status]     Thinking (iteration 1)...")
    print("  [text_delta] Let me look that up")
    print("  [text_delta]  for you.\n")
    print("  [tool_call]  weather_api(query='Tokyo')")
    print("  [tool_result] 22°C, Partly Cloudy\n")
    print("  [status]     Thinking (iteration 2)...")
    print("  [text_delta] Tokyo is currently")
    print("  [text_delta]  22°C and partly cloudy.\n")
    print("  [tool_call]  calculate(expression='100/7')")
    print("  [tool_result] 14.285714285714286\n")
    print("  [status]     Thinking (iteration 3)...")
    print("  [text_delta] And 100 / 7 ≈ 14.29.\n")

    # --- Key implementation details ---
    print("=== Implementation Details ===\n")
    print("run_streaming() differs from run() in two key ways:\n")
    print("1. Uses provider.stream() instead of send_message()")
    print("   - Yields ResponseChunk objects one at a time")
    print("   - Text deltas yield immediately as text_delta events\n")
    print("2. Tool calls are buffered from streaming chunks")
    print("   - Each tool_call_id gets a buffer for name + arguments")
    print("   - Arguments arrive as partial JSON fragments")
    print("   - Only dispatched once the stream emits the complete call")
    print("   - Then tool results are fed back like normal\n")
    print("This pattern ensures:")
    print("  - No 10-second silence (text streams immediately)")
    print("  - Correct tool execution (calls buffered until complete)")
    print("  - Same event types for consumers (text, tool_call, etc.)")
    print("  - Plus new text_delta type for real-time character output")


if __name__ == "__main__":
    main()
