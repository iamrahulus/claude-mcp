"""MCP server: weather + business-customer tools.

Run standalone for inspection:  mcp dev server.py
The client launches it as a stdio subprocess; nothing here talks to the model.

Design notes (the exam-relevant bits):
- Tools return *structured* errors via CallToolResult(is_error=True), not bare strings.
  (Wire format is still `isError`; the v2 Python SDK exposes it as `is_error`.)
- The server validates its own inputs. Policy (sequencing/governance) lives in the
  CLIENT/orchestrator, because the server cannot see conversation history.
- stdout is the protocol channel on stdio transport: NEVER print() here. Log to stderr.
"""
import json
import os
import sys

import httpx
from mcp.server import MCPServer  # mcp 2.x: FastMCP was renamed MCPServer
from mcp.types import CallToolResult, TextContent

mcp = MCPServer("weather-and-customers")

# WEATHER_MODE=mock makes the server deterministic and offline (used by smoke_test.py).
WEATHER_MODE = os.environ.get("WEATHER_MODE", "live")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MOCK_CITIES = {
    "melbourne": ("Melbourne", "Australia", -37.8136, 144.9631),
    "sydney": ("Sydney", "Australia", -33.8688, 151.2093),
    "london": ("London", "United Kingdom", 51.5072, -0.1276),
}

# Mock "CRM". Two account types so the business-only rule has something to bite on.
CUSTOMERS = {
    "C-1001": {"customer_id": "C-1001", "name": "Harbour Logistics Pty Ltd", "acc_type": "business"},
    "C-1002": {"customer_id": "C-1002", "name": "Priya Nair", "acc_type": "personal"},
}
BUSINESS_ACCOUNTS = {
    "C-1001": {"customer_id": "C-1001", "abn": "51 824 753 556", "credit_limit_aud": 250000, "status": "active"},
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tool_error(category: str, message: str, retryable: bool, attempted_input: dict) -> CallToolResult:
    """Structured error. `retryable` is the *tool's* claim; the orchestrator decides what to do with it."""
    payload = {
        "errorCategory": category,
        "isRetryable": retryable,
        "message": message,
        "attemptedInput": attempted_input,
    }
    return CallToolResult(is_error=True, content=[TextContent(type="text", text=json.dumps(payload))])


def ok(payload: dict) -> CallToolResult:
    return CallToolResult(is_error=False, content=[TextContent(type="text", text=json.dumps(payload))])


@mcp.tool()
async def geocode_city(city: str) -> CallToolResult:
    """Resolve a city name to latitude/longitude. Use this BEFORE get_weather when the user
    gave a place name rather than coordinates. Input: a city name like 'Melbourne'.
    Returns name, country, latitude, longitude. Not for street addresses."""
    args = {"city": city}
    if not city or not city.strip():
        return tool_error("invalid_input", "city must be a non-empty string", False, args)

    if WEATHER_MODE == "mock":
        hit = MOCK_CITIES.get(city.strip().lower())
        if not hit:
            return tool_error("not_found", f"No match for '{city}'", False, args)
        name, country, lat, lon = hit
        return ok({"name": name, "country": country, "latitude": lat, "longitude": lon})

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(GEOCODE_URL, params={"name": city, "count": 1})
            r.raise_for_status()
            results = r.json().get("results") or []
    except httpx.TimeoutException:
        return tool_error("timeout", "Geocoding service timed out", True, args)
    except httpx.HTTPError as e:
        return tool_error("upstream_unavailable", f"Geocoding request failed: {type(e).__name__}", True, args)

    if not results:
        return tool_error("not_found", f"No match for '{city}'", False, args)
    h = results[0]
    return ok({"name": h["name"], "country": h.get("country"), "latitude": h["latitude"], "longitude": h["longitude"]})


@mcp.tool()
async def get_weather(latitude: float, longitude: float) -> CallToolResult:
    """Get current weather for a coordinate pair. Inputs: latitude (-90..90) and longitude
    (-180..180) as decimal degrees. Returns temperature in Celsius and wind speed in km/h.
    Requires coordinates, not a city name; call geocode_city first if you only have a name."""
    args = {"latitude": latitude, "longitude": longitude}
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return tool_error("invalid_input", "latitude must be in [-90,90] and longitude in [-180,180]", False, args)

    if WEATHER_MODE == "mock":
        return ok({"temperature_c": 18.4, "wind_kmh": 11.0, "source": "mock"})

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(FORECAST_URL, params={
                "latitude": latitude, "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
            })
            r.raise_for_status()
            cur = r.json()["current"]
    except httpx.TimeoutException:
        return tool_error("timeout", "Weather service timed out", True, args)
    except httpx.HTTPError as e:
        return tool_error("upstream_unavailable", f"Weather request failed: {type(e).__name__}", True, args)
    return ok({"temperature_c": cur["temperature_2m"], "wind_kmh": cur["wind_speed_10m"], "source": "open-meteo"})


@mcp.tool()
async def get_customer(customer_id: str) -> CallToolResult:
    """Look up a customer profile by ID (format 'C-1234'). Returns name and acc_type
    ('business' or 'personal'). Read-only. Use this to find out what kind of account a
    customer has BEFORE asking for account details."""
    args = {"customer_id": customer_id}
    rec = CUSTOMERS.get(customer_id)
    if rec is None:
        return tool_error("not_found", f"No customer with id {customer_id}", False, args)
    return ok(rec)


@mcp.tool()
async def get_business_account(customer_id: str) -> CallToolResult:
    """Get business account details (ABN, credit limit, status) for a customer. Only valid
    for customers whose acc_type is 'business'; call get_customer first to confirm that.
    Read-only."""
    args = {"customer_id": customer_id}
    rec = BUSINESS_ACCOUNTS.get(customer_id)
    if rec is None:
        # Server-side backstop: even if the orchestrator's policy were bypassed, a personal
        # customer still gets a clean, structured refusal rather than data.
        return tool_error("not_applicable", f"{customer_id} has no business account", False, args)
    return ok(rec)


if __name__ == "__main__":
    mcp.run(transport="stdio")