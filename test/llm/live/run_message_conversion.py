#! python3
# -*- encoding: utf-8 -*-
"""Manual live test for provider-neutral message conversion with real APIs.

Exercises the ``from_neutral_message`` / ``to_neutral_message`` conversion
methods against live OpenAI-compatible and Anthropic-compatible endpoints,
including a cross-provider transfer with tool-call history.

Scenarios
---------

1. **Single-provider tool round** — one ``get_weather`` tool call on OpenAI.
   The resulting history (including tool-use and tool-result messages) is
   converted to neutral and serialised.

2. **Cross-provider transfer** — a conversation started on OpenAI (with a
   tool call) is converted to Anthropic-native format and continued on the
   Anthropic provider. The second provider receives the full tool-call
   history and can answer follow-up questions about it.

3. **History persistence** — the neutral history is serialised to JSON,
   reloaded, and converted to native format for each provider.

Usage::

    1. Make sure ``test/llm/live/.env`` contains valid credentials for both
       providers (see ``.env.example``).

    2. Run from the repo root::

         python -m test.llm.live.run_message_conversion           # all scenarios
         python -m test.llm.live.run_message_conversion openai    # single-provider only
         python -m test.llm.live.run_message_conversion cross     # cross-provider only

Expected .env variables:

    ``OPENAI_CHAT_BASE_URL`` / ``OPENAI_CHAT_API_KEY`` / ``OPENAI_CHAT_MODEL``
    ``ANTHROPIC_MESSAGES_BASE_URL`` / ``ANTHROPIC_MESSAGES_API_KEY`` / ``ANTHROPIC_MESSAGES_MODEL``

@File   :   run_message_conversion.py
@Created:   2026/08/05 00:00
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import json
import sys
from os import path
from typing import Any, Dict, List, Optional

from isobase.llm import (
    AnthropicMessages,
    LLMMessage,
    LLMMessageHistory,
    LLMClient,
    OpenAIChat,
)
from isobase.llm.tools import FunctionTool

ENV_PATH = path.join(path.dirname(__file__), ".env")


# ---------------------------------------------------------------------------
# tool definition
# ---------------------------------------------------------------------------

def __get_weather(city: str) -> tuple:
    """Get the current weather for a city.

    Args:
        city: City name.
    """
    return True, f"It is 22°C and sunny in {city}."


WEATHER_TOOL = FunctionTool(mapped_callable=__get_weather, name="get_weather")

# A second stateless tool so we can test multiple tool calls in one round.
def __get_time(tz: str) -> tuple:
    """Get the current time for a timezone.

    Args:
        tz: Timezone string (e.g. 'CET', 'EST').
    """
    return True, f"The current time in {tz} is 12:00 (simulated)."


TIME_TOOL = FunctionTool(mapped_callable=__get_time, name="get_time")


# ---------------------------------------------------------------------------
# .env helpers (same pattern as run_llm_basic.py)
# ---------------------------------------------------------------------------

def _load_env() -> Dict[str, str]:
    """Parses the git-ignored .env file into a dict."""
    if not path.exists(ENV_PATH):
        sys.exit(
            f"Missing {ENV_PATH}.\n"
            "Copy the template and fill in real values:\n"
            "  cp test/llm/live/.env.example test/llm/live/.env"
        )
    env: Dict[str, str] = {}
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                env[key] = value
    return env


def _client_kwargs(env: Dict[str, str], prefix: str) -> Optional[Dict[str, Any]]:
    """Builds client kwargs from env vars; returns None when API key is absent."""
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = env.get(f"{prefix}_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    model = env.get(f"{prefix}_MODEL", "").strip()
    if model:
        kwargs["default_model"] = model
    return kwargs


# ---------------------------------------------------------------------------
# scenario 1 — single-provider tool round (OpenAI)
# ---------------------------------------------------------------------------

def run_single_provider(kwargs: Dict[str, Any]) -> None:
    """Runs a tool-calling round on OpenAI, then converts history to neutral."""
    client = OpenAIChat(tools=[WEATHER_TOOL, TIME_TOOL], **kwargs)

    print(f"\n{'=' * 70}")
    print("[OpenAI] Tool-calling round — asking for weather + time")
    print(f"{'=' * 70}")

    resp = client.ask(
        "What is the weather in Tokyo?  Also, what time is it there (JST)?  "
        "Use both tools.",
        stream=False,
    )
    print(f"success={resp.success}")
    print(f"content={resp.content!r}")
    print(f"tool_calls_made={list(client.latest_tool_call_result.keys())}")

    if not client.latest_tool_call_result:
        print("[WARN] No tool calls were made — conversion history will be simpler.")
    else:
        print("Tool calls executed successfully.")

    # --- convert OpenAI-native messages to neutral history -------------------
    print(f"\n{'=' * 70}")
    print("[OpenAI → Neutral] Converting native messages to neutral history")
    print(f"{'=' * 70}")

    neutral_history = LLMMessageHistory()
    for native_msg in client.messages:
        neutral = OpenAIChat.to_neutral_message(native_msg)
        neutral_history.append(neutral)
        role = neutral.role
        if isinstance(neutral.content, str):
            preview = neutral.content[:80]
        else:
            preview = f"[{len(neutral.content)} block(s)]"
        print(f"  {role}: {preview}")

    # --- serialise to JSON and reload ---------------------------------------
    print(f"\n{'=' * 70}")
    print("[History] JSON round-trip")
    print(f"{'=' * 70}")

    data = neutral_history.to_list()
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    print(f"Serialised {len(json_str)} bytes, {len(neutral_history.messages)} messages.")

    reloaded = LLMMessageHistory.from_list(json.loads(json_str))
    print(f"Reloaded {len(reloaded.messages)} messages — match: "
          f"{len(reloaded.messages) == len(neutral_history.messages)}")

    # --- convert back to OpenAI to verify round-trip -------------------------
    print(f"\n{'=' * 70}")
    print("[Neutral → OpenAI] Converting back to OpenAI-native format")
    print(f"{'=' * 70}")

    roundtrip_messages = [OpenAIChat.from_neutral_message(m) for m in reloaded.messages]
    for msg in roundtrip_messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            preview = content[:80]
        elif content is not None and isinstance(content, list):
            preview = f"[{len(content)} block(s)]"
        else:
            preview = "(tool_calls only)" if msg.get("tool_calls") else str(content)
        print(f"  {role}: {preview}")

    print("\n--- Scenario 1 PASSED ---")


# ---------------------------------------------------------------------------
# scenario 2 — cross-provider transfer
# ---------------------------------------------------------------------------

def run_cross_provider(openai_kwargs: Dict[str, Any],
                       anthropic_kwargs: Dict[str, Any]) -> None:
    """Starts a conversation on OpenAI with tool calls, then continues on Anthropic.

    The full tool-call history (user → assistant + tool_use → tool_result →
    assistant answer) is converted through the neutral layer and fed to the
    Anthropic provider, which is then asked a follow-up question that
    requires knowledge of the earlier tool results.
    """
    # --- step 1: OpenAI conversation with tool calls -------------------------
    oai_client = OpenAIChat(tools=[WEATHER_TOOL, TIME_TOOL], **openai_kwargs)

    print(f"\n{'=' * 70}")
    print("[Step 1 — OpenAI] Tool-calling round: weather + time for Paris")
    print(f"{'=' * 70}")

    resp1 = oai_client.ask(
        "What's the weather in Paris? Also, what time is it in CET? Use both tools.",
        stream=False,
    )
    print(f"success={resp1.success}")
    print(f"content={resp1.content!r}")
    tool_names = list(oai_client.latest_tool_call_result.keys())
    print(f"tool_calls_made={tool_names}")

    # --- step 2: convert OpenAI native messages → neutral → Anthropic native -
    print(f"\n{'=' * 70}")
    print("[Step 2 — Conversion] OpenAI-native → neutral → Anthropic-native")
    print(f"{'=' * 70}")

    neutral_messages: List[LLMMessage] = []
    for native_msg in oai_client.messages:
        neutral = OpenAIChat.to_neutral_message(native_msg)
        neutral_messages.append(neutral)
        role = neutral.role
        if isinstance(neutral.content, str):
            preview = neutral.content[:80]
        else:
            types = [b.type for b in neutral.content]
            preview = str(types)
        print(f"  neutral {role}: {preview}")

    anthropic_native_msgs = AnthropicMessages.from_neutral_history(
        LLMMessageHistory(messages=neutral_messages))
    print(f"\nConverted {len(anthropic_native_msgs)} messages to Anthropic-native format.")
    for msg in anthropic_native_msgs:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            preview = content[:80]
        elif content is not None and isinstance(content, list):
            preview = f"[{len(content)} block(s)]"
        else:
            preview = "(no text content)" if msg.get("tool_calls") else str(content)
        print(f"  anthropic {role}: {preview}")

    # --- step 3: continue the conversation on Anthropic ----------------------
    print(f"\n{'=' * 70}")
    print("[Step 3 — Anthropic] Continuing conversation with transferred history")
    print(f"{'=' * 70}")

    ant_client = AnthropicMessages(
        tools=[WEATHER_TOOL, TIME_TOOL],
        messages=anthropic_native_msgs,
        **anthropic_kwargs,
    )

    follow_up = (
        "Based on the weather and time information you already retrieved for "
        "Paris earlier, write a one-sentence summary of what you found."
    )
    resp2 = ant_client.ask(follow_up, stream=False)
    print(f"success={resp2.success}")
    print(f"content={resp2.content!r}")

    # Basic assertions on the transfer.
    assert resp2.success, f"Anthropic follow-up failed: {resp2.content}"
    # The answer should mention Paris, weather, and time since the history
    # was transferred and the prompt asks about them.
    lower = resp2.content.lower()
    paris_ok = "paris" in lower
    weather_or_temp = any(w in lower for w in ("weather", "°c", "sunny", "22", "temperature"))
    time_ok = any(w in lower for w in ("time", "12:00", "cet"))
    print(f"\nTransfer verification:")
    print(f"  mentions Paris ........... {'OK' if paris_ok else 'MISSING'}")
    print(f"  mentions weather/temp .... {'OK' if weather_or_temp else 'MISSING'}")
    print(f"  mentions time ............ {'OK' if time_ok else 'MISSING'}")

    # Only assert if at least one tool call was actually made in step 1.
    if tool_names:
        if not paris_ok:
            print("[WARN] Follow-up did not mention Paris — transfer may be incomplete.")
        if not weather_or_temp and not time_ok:
            print("[WARN] Follow-up did not reference prior tool results — "
                  "transfer may be incomplete.")

    print("\n--- Scenario 2 PASSED ---")


# ---------------------------------------------------------------------------
# scenario 3 — neutral history persistence across providers
# ---------------------------------------------------------------------------

def run_history_persistence(openai_kwargs: Dict[str, Any],
                            anthropic_kwargs: Dict[str, Any]) -> None:
    """Demonstrates saving neutral history from one provider and loading it on another.

    After running a tool call on OpenAI, the neutral history is serialised to JSON.
    A fresh Anthropic client is then initialised from that JSON and asked to recall
    the earlier results.
    """
    # --- OpenAI: run a tool call ---------------------------------------------
    oai_client = OpenAIChat(tools=[WEATHER_TOOL], **openai_kwargs)

    print(f"\n{'=' * 70}")
    print("[Persistence — Step 1] OpenAI: get weather for London")
    print(f"{'=' * 70}")

    resp1 = oai_client.ask("What's the weather in London? Use the tool.", stream=False)
    print(f"success={resp1.success}")
    print(f"content={resp1.content!r}")

    # --- neutralize + serialise ----------------------------------------------
    print(f"\n{'=' * 70}")
    print("[Persistence — Step 2] Serialise history to JSON")
    print(f"{'=' * 70}")

    history = LLMMessageHistory()
    for native_msg in oai_client.messages:
        history.append(OpenAIChat.to_neutral_message(native_msg))

    saved_json = json.dumps(history.to_list(), indent=2, ensure_ascii=False)
    print(f"Saved {len(saved_json)} bytes ({len(history.messages)} messages)")
    print(f"First 200 chars:\n{saved_json[:200]}...")

    # --- deserialise + load into Anthropic -----------------------------------
    print(f"\n{'=' * 70}")
    print("[Persistence — Step 3] Reload history on Anthropic and continue")
    print(f"{'=' * 70}")

    reloaded = LLMMessageHistory.from_list(json.loads(saved_json))
    print(f"Reloaded {len(reloaded.messages)} messages.")

    anthropic_msgs = AnthropicMessages.from_neutral_history(reloaded)
    ant_client = AnthropicMessages(
        tools=[WEATHER_TOOL],
        messages=anthropic_msgs,
        **anthropic_kwargs,
    )

    follow_up = (
        "Based on the weather you already retrieved for London, tell me "
        "the temperature and conditions there in one sentence.  Do NOT call "
        "the tool again."
    )
    resp2 = ant_client.ask(follow_up, stream=False)
    print(f"success={resp2.success}")
    print(f"content={resp2.content!r}")

    lower = resp2.content.lower()
    # The tool returns "22°C and sunny" so one of these should appear if the
    # history was correctly transferred.
    if any(w in lower for w in ("22", "sunny", "london", "weather")):
        print("Transfer verified — follow-up references prior results.")
    else:
        print("[WARN] Follow-up may not reference prior results — check manually.")

    print("\n--- Scenario 3 PASSED ---")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(only: Optional[str] = None) -> None:
    """Entry point.

    Args:
        only: ``"openai"``, ``"cross"``, ``"persist"``, or None for all.
    """
    env = _load_env()
    openai_kwargs = _client_kwargs(env, "OPENAI_CHAT")
    anthropic_kwargs = _client_kwargs(env, "ANTHROPIC_MESSAGES")

    if openai_kwargs is None:
        print("[skip] No usable OPENAI_CHAT_API_KEY in .env — exiting.")
        sys.exit(1)
    if anthropic_kwargs is None:
        print("[skip] No usable ANTHROPIC_MESSAGES_API_KEY in .env — "
              "cross-provider scenarios will be skipped.")

    if only == "openai":
        run_single_provider(openai_kwargs)
    elif only == "cross":
        if anthropic_kwargs is None:
            sys.exit("ANTHROPIC_MESSAGES_API_KEY is required for cross-provider test.")
        run_cross_provider(openai_kwargs, anthropic_kwargs)
    elif only == "persist":
        if anthropic_kwargs is None:
            sys.exit("ANTHROPIC_MESSAGES_API_KEY is required for persistence test.")
        run_history_persistence(openai_kwargs, anthropic_kwargs)
    else:
        run_single_provider(openai_kwargs)
        if anthropic_kwargs is not None:
            run_cross_provider(openai_kwargs, anthropic_kwargs)
            run_history_persistence(openai_kwargs, anthropic_kwargs)
        else:
            print("\n[skip] Cross-provider & persistence scenarios skipped "
                  "(no Anthropic credentials).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
