"""MCP (Model Context Protocol) client support for glm-rev.

Port of claude-rev's mcp.py. Bridges the official `mcp` SDK into this
synchronous codebase:

* Server discovery: standard `~/.mcp.json` and `<cwd>/.mcp.json` files
  (`mcpServers` format — stdio servers via command/args/env, remote
  servers via type "http"/url).
* Each connected server runs in its own background thread with a private
  asyncio event loop; blocking calls are bridged to the loop with
  `asyncio.run_coroutine_threadsafe` so the rest of the client stays sync.
* `MCPManager` exposes a tool registry (collision-safe effective names),
  `call_tool`, `status`, and a prompt-contract block describing every tool.
"""

import asyncio
import json
import os
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .config import LOCAL_TOOL_NAMES

CONFIG_FILES = (
    os.path.expanduser("~/.mcp.json"),
    os.path.join(os.getcwd(), ".mcp.json"),
)
HTTP_TYPES = ("http", "streamablehttp", "httpstream", "sse")
CONNECT_TIMEOUT = 30
CALL_TIMEOUT = 120


def load_mcp_config():
    """Merge MCP server config from the standard files. Returns {name: entry}."""
    servers = {}
    for path in CONFIG_FILES:
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path))
        except Exception:
            continue
        found = data.get("mcpServers")
        if isinstance(found, dict):
            servers.update(found)
    return servers


def is_http_entry(entry):
    if not isinstance(entry, dict):
        return False
    return entry.get("type", "").lower() in HTTP_TYPES or "url" in entry


def redact_url(url):
    """Redact credentials from a server URL for display (keep scheme://host)."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.hostname}" + (f":{p.port}" if p.port else "")
    except Exception:
        pass
    return url


def _format_result(result):
    """Aggregate an MCP CallToolResult into a plain-text string."""
    parts = []
    for block in result.content or []:
        txt = getattr(block, "text", None)
        if txt is not None:
            parts.append(txt)
    if result.structuredContent is not None and not parts:
        parts.append(json.dumps(result.structuredContent, ensure_ascii=False))
    text = "\n".join(parts).strip() or "(no output)"
    if getattr(result, "isError", False):
        return "ERROR: " + text
    return text


class _Server:
    """One connected MCP server (stdio or streamable HTTP)."""

    def __init__(self, name, entry):
        self.name = name
        self.entry = entry
        self.tools = []
        self.error = None
        self.connected = False
        self._loop = None
        self._session = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    # -- lifecycle ----------------------------------------------------

    def start(self):
        self._thread = threading.Thread(
            target=self._run_loop, name=f"mcp-{self.name}", daemon=True)
        self._thread.start()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._amain())
        except Exception as e:  # noqa: BLE001 — surface to status()
            self.error = e
            self._ready.set()

    async def _amain(self):
        streams = None
        session = None
        try:
            if is_http_entry(self.entry):
                streams = streamable_http_client(self.entry["url"], terminate_on_close=True)
                read, write, _ = await streams.__aenter__()
            elif self.entry.get("command"):
                params = StdioServerParameters(
                    command=self.entry["command"],
                    args=list(self.entry.get("args") or []),
                    env=self.entry.get("env"),
                    cwd=self.entry.get("cwd"),
                )
                streams = stdio_client(params)
                read, write = await streams.__aenter__()
            else:
                raise RuntimeError("no 'url' or 'command' in mcp server config")
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            self._session = session
            result = await session.list_tools()
            self.tools = list(result.tools or [])
            self.connected = True
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.2)
        finally:
            if session is not None:
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    pass
            if streams is not None:
                try:
                    await streams.__aexit__(None, None, None)
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    # -- calls ---------------------------------------------------------

    def call_tool(self, name, arguments):
        if self._session is None or not self.connected:
            return "ERROR: mcp server not connected"
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments or {}), self._loop)
        result = fut.result(timeout=CALL_TIMEOUT)
        return _format_result(result)


class MCPManager:
    """Connects MCP servers and routes tool calls by effective name."""

    def __init__(self, config=None):
        self.config = config if config is not None else load_mcp_config()
        self.servers = {}
        self.registry = {}

    # -- lifecycle ----------------------------------------------------

    def start(self, names=None):
        """Connect the given servers (default: all configured)."""
        if not names:
            names = list(self.config)
        missing = [n for n in names if n not in self.config]
        for name in missing:
            self.servers[name] = _Server(name, {})
        for name in names:
            if name in self.config and name not in self.servers:
                self.servers[name] = _Server(name, self.config[name])
                self.servers[name].start()
        for s in self.servers.values():
            if s._thread is not None:
                s._ready.wait(timeout=CONNECT_TIMEOUT)
        self._build_registry()

    def stop(self):
        for s in self.servers.values():
            s.stop()
        self.servers.clear()
        self.registry = {}

    def _build_registry(self):
        self.registry = {}
        for sname, s in self.servers.items():
            for tool in s.tools:
                ename = tool.name
                if ename in self.registry or ename in LOCAL_TOOL_NAMES:
                    ename = f"{sname}_{tool.name}"
                self.registry[ename] = (sname, tool)

    # -- queries --------------------------------------------------------

    def connected_count(self):
        return sum(1 for s in self.servers.values() if s.connected)

    def tool_count(self):
        return len(self.registry)

    def status_lines(self):
        out = []
        for name in sorted(self.config):
            s = self.servers.get(name)
            if s is None:
                out.append(f"  {name:24s} not started")
            elif s.error:
                out.append(f"  {name:24s} ERROR: {s.error}")
            elif s.connected:
                out.append(f"  {name:24s} connected, {len(s.tools)} tool(s)")
            else:
                out.append(f"  {name:24s} connecting...")
        return out

    def server_of(self, tool_name):
        hit = self.registry.get(tool_name)
        return hit[0] if hit else None

    # -- dispatch ---------------------------------------------------------

    def call_tool(self, effective_name, arguments):
        """Route a call by effective tool name. Returns (ok, text)."""
        hit = self.registry.get(effective_name)
        if not hit:
            return False, f"mcp tool not found: {effective_name}"
        sname, tool = hit
        s = self.servers[sname]
        if s.error:
            return False, f"mcp server {sname} failed: {s.error}"
        try:
            text = s.call_tool(tool.name, arguments)
            return (text.startswith("ERROR: ") is False), text
        except Exception as e:
            return False, f"mcp call failed ({sname}.{tool.name}): {e}"

    def contract_block(self, tool_cap=320):
        """Prompt-contract appendix describing every MCP tool (effective names)."""
        lines = []
        for sname in sorted(self.servers):
            s = self.servers[sname]
            if not s.connected:
                continue
            lines.append(f'Additional tools are available through the MCP server '
                         f'"{sname}". Call them with the exact same one-line format.')
            for tool in s.tools:
                ename = self._effective_name(sname, tool.name)
                schema = tool.inputSchema or {}
                props = schema.get("properties") or {}
                req = set(schema.get("required") or [])
                compact = ", ".join(
                    f"{k}:<{'req' if k in req else 'opt'}>" for k in props) or "(no args)"
                desc = (tool.description or "").strip().replace("\n", " ")
                line = f'  TOOL:{ename}({json.dumps(schema)[:120] if props else "{}"})'
                if len(line) > tool_cap:
                    line = line[:tool_cap] + "..."
                lines.append(line)
                if desc:
                    lines.append(f"      -> {desc[:tool_cap]}")
                if compact and compact != "(no args)":
                    lines.append(f"      args: {compact}")
            lines.append("")
        return "\n".join(lines)

    def _effective_name(self, sname, tool_name):
        ename = tool_name
        if ename in LOCAL_TOOL_NAMES:
            ename = f"{sname}_{tool_name}"
        for s, t in self.registry.values():
            if t.name == tool_name and s != sname:
                ename = f"{sname}_{tool_name}"
                break
        return ename

    def tool_names(self, server=None):
        names = set()
        for sname, s in self.servers.items():
            if server and sname != server:
                continue
            for tool in s.tools:
                names.add(self._effective_name(sname, tool.name))
        return sorted(names)