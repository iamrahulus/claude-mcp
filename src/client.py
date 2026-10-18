"""MCP client + governed agent loop.

Three layers, deliberately separate:
  1. MCP session   : discovery (list_tools) and execution (call_tool) over stdio.
  2. Policy        : decides whether a proposed call may run. Lives HERE, not in the server,
                     because only the orchestrator sees conversation history.
  3. Agent loop    : the stop_reason == "tool_use" loop with a MAX_ITERATIONS guard.

Usage:  python client.py "What's the weather in Melbourne?"
Needs ANTHROPIC_API_KEY in a project-scoped .env (python-dotenv). Never a shell env var.
"""
import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-haiku-4-5"
MAX_ITERATIONS = 8
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 1. MCP session
@asynccontextmanager
async def mcp_session(env: dict | None = None):
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(HERE, "server.py")],
        env={**os.environ, **(env or {})},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def to_anthropic_tools(mcp_tools, allowed: set[str] | None = None) -> list[dict]:
    """MCP -> Anthropic. Same JSON Schema, near-identical key: MCP's wire name is inputSchema
    (v2 Python attr: input_schema), Anthropic's is input_schema.
    `allowed` is the tool manifest: the model never even SEES tools outside it."""
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in mcp_tools
        if allowed is None or t.name in allowed
    ]


# ---------------------------------------------------------------- 2. Policy
class Policy:
    """Outcome rules, not workflow sequences: 'coordinates must be trusted', 'account must be
    known-business', rather than 'geocoder must have run'. State is per-conversation and is
    derived from *successful* prior results, never from what the model claims."""

    def __init__(self):
        self.trusted_coords: list[tuple[float, float]] = []
        self.business_customers: set[str] = set()
        self.known_customers: dict[str, str] = {}

    def check(self, name: str, args: dict) -> str | None:
        """Return None to allow, or a rejection message to feed back to the model."""
        if name == "get_weather":
            pair = (args.get("latitude"), args.get("longitude"))
            if pair not in self.trusted_coords:
                return ("POLICY: coordinates were not produced by geocode_city in this conversation "
                        "and were not supplied by the user. Call geocode_city first.")
        if name == "get_business_account":
            cid = args.get("customer_id")
            if cid not in self.known_customers:
                return f"POLICY: call get_customer for {cid} first to establish the account type."
            if cid not in self.business_customers:
                return f"POLICY: {cid} is not a business customer; business account details are not available."
        return None

    def observe(self, name: str, args: dict, result: dict) -> None:
        """Update state from a SUCCESSFUL result only."""
        if name == "geocode_city":
            self.trusted_coords.append((result["latitude"], result["longitude"]))
        elif name == "get_customer":
            self.known_customers[result["customer_id"]] = result["acc_type"]
            if result["acc_type"] == "business":
                self.business_customers.add(result["customer_id"])

    def trust_user_supplied(self, user_text: str, lat: float, lon: float) -> None:
        """Explicit escape hatch: coords typed by the user are trusted (matches your Exercise 5-6 rule)."""
        self.trusted_coords.append((lat, lon))


def parse_result(call_result) -> tuple[bool, dict]:
    """MCP CallToolResult -> (is_error, parsed_json). Our server always emits JSON text."""
    text = "".join(c.text for c in call_result.content if getattr(c, "type", "") == "text")
    try:
        return bool(call_result.is_error), json.loads(text)
    except json.JSONDecodeError:
        # A tool that violates the contract (bare string) still gets surfaced, never swallowed.
        return bool(call_result.is_error), {"errorCategory": "unstructured", "message": text}


# ---------------------------------------------------------------- audit (minimal)
def audit(events: list, correlation_id: str, event_type: str, **fields) -> None:
    """In-memory stand-in for your unified events table. Swap for your AuditLogger."""
    events.append({"correlation_id": correlation_id, "event_type": event_type, **fields})


# ---------------------------------------------------------------- 3. One tool_use block -> one tool_result block
async def execute_tool_use(session, policy: Policy, events: list, cid: str, block) -> dict:
    name, args, tid = block.name, block.input, block.id
    audit(events, cid, "tool_call", tool_use_id=tid, tool=name, args=args)

    rejection = policy.check(name, args)
    if rejection:
        # Governance event, NOT a reliability failure: goes back to the model, never to retry.
        audit(events, cid, "policy_rejection", tool_use_id=tid, tool=name, message=rejection)
        return {"type": "tool_result", "tool_use_id": tid, "content": rejection, "is_error": True}

    call_result = await session.call_tool(name, args)
    is_error, payload = parse_result(call_result)
    audit(events, cid, "tool_result", tool_use_id=tid, tool=name, is_error=is_error, payload=payload)

    if not is_error:
        policy.observe(name, args, payload)
    return {
        "type": "tool_result",
        "tool_use_id": tid,
        "content": json.dumps(payload),
        **({"is_error": True} if is_error else {}),
    }


# ---------------------------------------------------------------- the loop
async def run_agent(user_message: str, allowed_tools: set[str] | None = None) -> tuple[str, list]:
    from anthropic import AsyncAnthropic
    from dotenv import load_dotenv

    load_dotenv(os.path.join(HERE, ".env"))
    client = AsyncAnthropic()
    cid, events, policy = str(uuid.uuid4()), [], Policy()

    async with mcp_session() as session:
        tools = to_anthropic_tools((await session.list_tools()).tools, allowed_tools)
        messages = [{"role": "user", "content": user_message}]

        for i in range(MAX_ITERATIONS):
            resp = await client.messages.create(
                model=MODEL, max_tokens=1024, tools=tools, messages=messages,
            )
            print(f"[iter {i}] stop_reason={resp.stop_reason} in_tokens={resp.usage.input_tokens}", file=sys.stderr)

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                audit(events, cid, "final_answer", text=text)
                return text, events

            messages.append({"role": "assistant", "content": resp.content})
            # ALL results for this turn go back in ONE user message.
            results = [await execute_tool_use(session, policy, events, cid, b)
                       for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "user", "content": results})

        audit(events, cid, "loop_guard_tripped", iterations=MAX_ITERATIONS)
        return f"Stopped: exceeded {MAX_ITERATIONS} iterations.", events


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Melbourne right now?"
    answer, evs = asyncio.run(run_agent(q))
    print(answer)
    print(f"\n{len(evs)} audit events, correlation_id={evs[0]['correlation_id']}")