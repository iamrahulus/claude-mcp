"""Inspect the server on its own, no Node, no uv, no API key.
Launches server.py over stdio, lists every tool with its schema, then calls one.
Usage:  poetry run python list_tools.py
"""
import asyncio
import json

from client import mcp_session, parse_result


async def main() -> None:
    async with mcp_session({"WEATHER_MODE": "mock"}) as s:
        for t in (await s.list_tools()).tools:
            print(f"\n== {t.name}\n{t.description}\ninput_schema: {json.dumps(t.input_schema)}")
        print("\n-- calling geocode_city('Melbourne')")
        is_error, body = parse_result(await s.call_tool("geocode_city", {"city": "Melbourne"}))
        print(f"is_error={is_error} body={body}")


asyncio.run(main())