"""Smoke test: real MCP stdio protocol, no Anthropic API key, no model.
Fake tool_use blocks stand in for what the model would propose, so we test the plumbing
and the governance in isolation (test components before wiring them into the loop)."""
import asyncio
import importlib.metadata as md
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

# Find client.py whether the project is flat (everything in one folder) or split into src/ + tests/.
_here = Path(__file__).resolve().parent
for _cand in (_here, _here / "src", _here.parent / "src"):
    if (_cand / "client.py").exists():
        sys.path.insert(0, str(_cand))
        break

sys.stdout.reconfigure(line_buffering=True)
# Banner first, before anything that could fail: if you see nothing at all, this file isn't the one running.
print(f"smoke_test: python {sys.version.split()[0]} | mcp {md.version('mcp')} | {__file__}")

from client import Policy, execute_tool_use, mcp_session, parse_result, to_anthropic_tools  # noqa: E402

fails = []


def check(label, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(label)


def block(i, name, **args):
    return NS(id=f"toolu_{i}", name=name, input=args, type="tool_use")


async def main():
    async with mcp_session({"WEATHER_MODE": "mock"}) as s:
        # --- discovery
        tools = (await s.list_tools()).tools
        names = {t.name for t in tools}
        check("discovers 4 tools", names == {"geocode_city", "get_weather", "get_customer", "get_business_account"}, names)
        a = to_anthropic_tools(tools)
        check("mapped to Anthropic input_schema", all("input_schema" in t and t["input_schema"].get("type") == "object" for t in a))
        manifest = to_anthropic_tools(tools, {"get_weather", "geocode_city"})
        check("manifest hides tools from the model", {t["name"] for t in manifest} == {"get_weather", "geocode_city"})

        # --- weather chain
        p, ev, cid = Policy(), [], "corr-1"
        r = await execute_tool_use(s, p, ev, cid, block(1, "get_weather", latitude=-37.8, longitude=144.9))
        check("fabricated coords rejected by policy", r.get("is_error") and r["content"].startswith("POLICY"))
        check("rejection never reached the server", not any(e["event_type"] == "tool_result" for e in ev))

        r = await execute_tool_use(s, p, ev, cid, block(2, "geocode_city", city="Melbourne"))
        check("geocode ok", not r.get("is_error") and '"latitude": -37.8136' in r["content"], r)
        r = await execute_tool_use(s, p, ev, cid, block(3, "get_weather", latitude=-37.8136, longitude=144.9631))
        check("weather allowed after geocode", not r.get("is_error") and "temperature_c" in r["content"], r)

        # --- structured errors from the server (policy bypassed on purpose: server must defend itself)
        err, body = parse_result(await s.call_tool("get_weather", {"latitude": 999, "longitude": 0}))
        check("server rejects out-of-range coords, structured", err and body["errorCategory"] == "invalid_input" and body["isRetryable"] is False and body["attemptedInput"]["latitude"] == 999, body)
        err, body = parse_result(await s.call_tool("geocode_city", {"city": "Atlantis"}))
        check("not_found is structured and non-retryable", err and body["errorCategory"] == "not_found" and body["isRetryable"] is False, body)

        # --- business customer chain
        p, ev = Policy(), []
        r = await execute_tool_use(s, p, ev, "corr-2", block(4, "get_business_account", customer_id="C-1001"))
        check("account details blocked before get_customer", r.get("is_error") and "get_customer" in r["content"])
        await execute_tool_use(s, p, ev, "corr-2", block(5, "get_customer", customer_id="C-1001"))
        r = await execute_tool_use(s, p, ev, "corr-2", block(6, "get_business_account", customer_id="C-1001"))
        check("business customer gets account after lookup", not r.get("is_error") and "credit_limit_aud" in r["content"], r)

        await execute_tool_use(s, p, ev, "corr-2", block(7, "get_customer", customer_id="C-1002"))
        r = await execute_tool_use(s, p, ev, "corr-2", block(8, "get_business_account", customer_id="C-1002"))
        check("personal customer blocked by policy", r.get("is_error") and "not a business customer" in r["content"])
        err, body = parse_result(await s.call_tool("get_business_account", {"customer_id": "C-1002"}))
        check("server backstop also refuses personal customer", err and body["errorCategory"] == "not_applicable", body)

        # --- audit shape
        types = [e["event_type"] for e in ev]
        check("audit has policy_rejection events", "policy_rejection" in types, types)
        check("every event carries the correlation id", all(e["correlation_id"] == "corr-2" for e in ev))

    # --- live mode: sandbox may have no route to Open-Meteo; that IS the structured-error path
    async with mcp_session({"WEATHER_MODE": "live"}) as s:
        err, body = parse_result(await s.call_tool("geocode_city", {"city": "Melbourne"}))
        print(f"INFO  live geocode -> isError={err} body={body}")
        if err:
            check("live failure is structured + retryable", body.get("errorCategory") in {"upstream_unavailable", "timeout"} and body.get("isRetryable") is True, body)

    print(f"\n{'ALL PASSED' if not fails else f'{len(fails)} FAILED: {fails}'}")
    return 1 if fails else 0


def run() -> int:
    """Never exit silently: any crash, including MCP's nested ExceptionGroups, is printed."""
    try:
        return asyncio.run(asyncio.wait_for(main(), timeout=120))
    except BaseException:
        print("\nSMOKE TEST CRASHED. If you see 'Connection closed', the server subprocess died on startup;"
              " run `python server.py` directly to see why.\n")
        traceback.print_exc(file=sys.stdout)
        return 2


if __name__ == "__main__":
    code = run()
    print(f"exit code {code}")
    sys.exit(code)