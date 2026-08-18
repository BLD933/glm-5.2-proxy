"""GLM web server: OpenAI-compatible API + multi-turn chat endpoint + web UI."""
import asyncio
import json
from contextlib import asynccontextmanager
import os
import threading
import time
import uuid
import uvicorn
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union

from glm_rev.solver import steal_captcha, CaptchaSolver, register_solver
from glm_rev.client import create_chat, refresh_token, build_features, stream_turn, fetch_reply_node
from glm_rev.config import (BASE, ENDPOINT, parse_tool_calls, strip_tool_lines,
                            REFUSAL_RE, TOOL_NUDGE_CLIENT, _intent_of)
from glm_rev.api import sign, headers, user_id_from_token
from glm_rev.tools import send_with_tools
from glm_rev.mcp import MCPManager, load_mcp_config
from glm_rev import captcha_aliyun as ca

HERE = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        raw_token = get_token()
        token = refresh_token(raw_token)
        register_solver(None, token)
        solver = _get_warm_solver()
        solver.start_background(token)
        print("[*] server lifespan: warm captcha solver started in background")
    except Exception as e:
        print(f"[!] server lifespan solver warmup failed: {e}")
    yield


app = FastAPI(title="GLM Proxy + UI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aliyun captcha tokens are SINGLE-USE — never cache or reuse them.
# A warm Playwright solver (daemon thread) issues a fresh token per request.
# Captcha solves are serialized on a threading.Lock because the warm solver is
# a single shared browser (used by both async endpoint code and tool threads).
_warm_solver = None
_solver_lock = threading.Lock()

# MCP servers are connected lazily on the first tools request.
_mcp = None
_mcp_lock = threading.Lock()

# Multi-turn sessions for /api/chat
SESSIONS = {}
MAX_HISTORY = 60
HISTORY_LIMIT = 120


def get_token() -> str:
    try:
        return open(os.path.join(HERE, "zai", "token.txt")).read().strip()
    except Exception:
        raise HTTPException(status_code=500, detail="ZAI token not found in zai/token.txt")


MODEL_ALIASES = {
    "gpt-4o": "glm-5.2", "gpt-4": "glm-5.2", "gpt-4-turbo": "glm-5.2",
    "claude-3-5-sonnet": "glm-5.2", "claude-sonnet-4": "glm-5.2",
    "deepseek-chat": "glm-5.2", "deepseek-reasoner": "glm-5.2",
    "default": "glm-5.2",
}


def _alias_models():
    return [{
        "id": a, "object": "model", "owned_by": "z.ai", "created": 0,
        "name": f"alias -> {t}",
    } for a, t in MODEL_ALIASES.items()]


def chunk_init(completion_id: str, created: int, model: str) -> dict:
    """First SSE chunk: declare assistant role before any content."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    }


def normalize_usage(u) -> Optional[dict]:
    """Map upstream usage (any shape) to OpenAI prompt/completion/total tokens."""
    if not isinstance(u, dict):
        return None
    p = u.get("prompt_tokens") or u.get("prompts") or u.get("in") or u.get("input_tokens")
    c = u.get("completion_tokens") or u.get("out") or u.get("output_tokens")
    if p is None and c is None:
        return None
    p = int(p or 0)
    c = int(c or 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def model_catalog():
    path = os.path.join(HERE, "zai", "models.json")
    try:
        data = json.load(open(path))

        def walk(o):
            if isinstance(o, dict):
                if o.get("openai") and o.get("id"):
                    yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)

        out, seen = [], set()
        for m in walk(data):
            mid = m.get("id")
            if mid in seen:
                continue
            seen.add(mid)
            out.append({
                "id": mid,
                "object": "model",
                "owned_by": "z.ai",
                "created": m.get("created", 0),
                "name": m.get("name", mid),
            })
        if out:
            return out + _alias_models()
    except Exception:
        pass
    fallback = [
        "glm-5.2", "GLM-5.1", "GLM-5-Turbo", "GLM-5v-Turbo", "glm-4.7",
        "glm-4.6v", "0727-106B-API", "0727-360B-API",
        "GLM-4.1V-Thinking-FlashX", "deep-research", "zero",
        "glm-4-flash", "0808-360B-DR", "glm-4-air-250414",
    ]
    return [{"id": m, "object": "model", "owned_by": "z.ai", "created": 0, "name": m} for m in fallback] + _alias_models()


class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[dict]]] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "glm-5.2"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 8192
    enable_thinking: Optional[bool] = True
    reasoning_effort: Optional[str] = "max"
    web_search: Optional[bool] = False
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    stream_options: Optional[dict] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


def _resolve_session_id(req: ChatCompletionRequest, request: Request) -> str:
    """Resolve session/conversation ID from request body, headers, or client host."""
    if getattr(req, "session_id", None):
        return str(req.session_id)
    if getattr(req, "conversation_id", None):
        return str(req.conversation_id)
    if getattr(req, "user", None):
        return str(req.user)
    for h in ("x-session-id", "x-conversation-id", "conversation_id", "session_id"):
        val = request.headers.get(h)
        if val:
            return val.strip()
    host = request.client.host if (request and request.client) else "default"
    if host in ("127.0.0.1", "::1", "localhost"):
        return f"v1_sess_{uuid.uuid4().hex[:16]}"
    return f"v1_sess_{host}"


def _commit_turn(session_id: str, prompt_text: str, answer_text: str, chat_id: str = None, last_ast_id: str = None):
    """Commit a completed conversational turn into the session cache."""
    if not session_id or not prompt_text:
        return
    sess = SESSIONS.setdefault(session_id, {"history": [], "chat_id": None, "last_assistant_id": None})
    hist = sess.setdefault("history", [])
    hist.append({"role": "user", "content": prompt_text})
    if answer_text:
        hist.append({"role": "assistant", "content": answer_text})
    if chat_id:
        sess["chat_id"] = chat_id
    if last_ast_id:
        sess["last_assistant_id"] = last_ast_id
    if len(hist) > HISTORY_LIMIT:
        sess["history"] = hist[-HISTORY_LIMIT:]


async def _fetch_reply(token, chat_id, cookie, after_user_id):
    """Resolve the server-assigned assistant reply node for the last streamed
    turn (GET /api/v1/chats/<id>, no captcha).

    Verified live 2026-08-16: chat.z.ai's completions backend recalls prior
    context ONLY when a request is parented at an assistant node id it
    streamed itself — POST /chats/<id>-rewritten ids are DEAF, and the SSE
    stream usually omits the stored reply id. Returns (assistant_id, None) or
    (None, None)."""
    return await asyncio.to_thread(fetch_reply_node, token, chat_id,
                                   after_user_id=after_user_id, cookie=cookie)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str
    model: str = "glm-5.2"
    enable_thinking: Optional[bool] = True
    reasoning_effort: Optional[str] = "max"
    web_search: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def _get_warm_solver() -> CaptchaSolver:
    global _warm_solver
    if _warm_solver is None:
        _warm_solver = CaptchaSolver()
    return _warm_solver


def _sync_captcha(token: str):
    """Synchronously acquire a fresh (captcha, cookie) pair. Thread-safe.

    Tries the in-memory Aliyun captcha pool first (no browser at all), then
    the warm Playwright solver (no browser relaunch per request); falls back
    to up to 3 one-shot headless solves. Serialized on a threading.Lock so
    the shared warm solver browser is never used concurrently. Raises
    RuntimeError if all attempts fail."""
    if ca.enabled():
        param = ca.solve()
        if param:
            return param, None
    with _solver_lock:
        solver = _get_warm_solver()
        try:
            if not solver._thread or not solver._thread.is_alive():
                captcha, cookie = solver.start(token)
            else:
                captcha, cookie = solver.solve(token)
            if solver._thread and solver._thread.is_alive():
                register_solver(solver, token)
            if captcha:
                return captcha, cookie
        except Exception as e:
            print(f"[!] Warm captcha solver unavailable: {e}")
        for attempt in range(2):
            try:
                captcha, cookie = asyncio.run(steal_captcha(token))
                if captcha:
                    return captcha, cookie
            except Exception as e:
                print(f"[!] One-shot captcha solver attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    raise RuntimeError("Failed to acquire Aliyun captcha verification token after multiple attempts")


async def acquire_captcha(token: str):
    """Async wrapper around _sync_captcha; keeps the event loop unblocked."""
    try:
        return await asyncio.to_thread(_sync_captcha, token)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _captcha_pair(token: str):
    """Tool-loop captcha_fn: returns (captcha, cookie) or None (no raise)."""
    try:
        return _sync_captcha(token)
    except Exception as e:
        print(f"[!] captcha acquisition failed: {e}")
        return None


def _get_mcp() -> Optional[MCPManager]:
    """Lazily connect configured MCP servers (thread-safe, once)."""
    global _mcp
    with _mcp_lock:
        if _mcp is None:
            try:
                cfg = load_mcp_config()
                if cfg:
                    m = MCPManager(cfg)
                    m.start()
                    if m.connected_count():
                        print(f"[*] MCP: {m.connected_count()} server(s), "
                              f"{m.tool_count()} tool(s)")
                    else:
                        print("[!] MCP: no servers connected")
                    _mcp = m
                else:
                    print("[*] no MCP servers configured (~/.mcp.json)")
            except Exception as e:
                print(f"[!] MCP unavailable ({e})")
                _mcp = None
    return _mcp


class _QueueWriter:
    """Pushes text from a tool-loop thread onto an asyncio.Queue as chunks."""

    def __init__(self, loop, queue):
        self._loop = loop
        self._q = queue

    def write(self, text):
        for i in range(0, len(text), 48):
            chunk = text[i:i + 48]
            self._loop.call_soon_threadsafe(self._q.put_nowait, chunk)


class _CaptureWriter:
    """Collects tool-loop final output (non-streaming requests)."""

    def __init__(self):
        self.parts = []

    def write(self, text):
        self.parts.append(text)

    @property
    def text(self):
        return "".join(self.parts)


def _assistant_tool_lines(m: Message) -> str:
    lines = []
    content = m.content
    if isinstance(content, list):
        content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
    if content:
        lines.append(content)
    for tc in m.tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name") or "?"
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        lines.append(f"TOOL:{name}({json.dumps(args or {}, ensure_ascii=False)})")
    return "\n".join(lines)


def _collect_directives(req: ChatCompletionRequest) -> str:
    parts = []
    for m in req.messages:
        if m.role not in ("system", "developer"):
            continue
        content = m.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _tool_messages(req: ChatCompletionRequest) -> list:
    """Sanitize OpenAI messages into GLM user/assistant {role, content} dicts.

    list-content (multimodal) is joined by text parts, role "tool" results are
    wrapped as user messages ("[Tool result for <id>]: ..."), and system/developer
    directives are collected and merged as a leading user message (GLM has no
    system role)."""
    out = []
    directives = _collect_directives(req)
    if directives:
        out.append({"role": "user", "content": directives})
    for m in req.messages:
        content = m.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        if m.role == "tool":
            tid = m.tool_call_id or m.name or "?"
            out.append({"role": "user", "content": f"[Tool result for {tid}]: {content}"})
        elif m.role == "user":
            out.append({"role": "user", "content": content or ""})
        elif m.role == "assistant":
            if m.tool_calls:
                out.append({"role": "assistant", "content": _assistant_tool_lines(m)})
            else:
                out.append({"role": "assistant", "content": content or ""})
    return out


def _tool_state(req: ChatCompletionRequest, token: str, session_id: str = None) -> dict:
    msgs = _tool_messages(req)
    last_user = -1
    for idx, mm in enumerate(msgs):
        if mm["role"] == "user":
            last_user = idx
    history = msgs[:last_user] if last_user >= 0 else []
    if session_id and len(req.messages) == 1 and not history:
        sess = SESSIONS.get(session_id)
        if sess and sess.get("history"):
            history = list(sess["history"])
    chat_id = None
    last_assistant_id = None
    last_assistant_parent_id = None
    seed_msg_id = None
    if session_id:
        sess = SESSIONS.get(session_id) or {}
        chat_id = sess.get("chat_id")
        last_assistant_id = sess.get("last_assistant_id")
        last_assistant_parent_id = sess.get("last_assistant_parent_id")
        seed_msg_id = sess.get("seed_msg_id")
    return {
        "token": token,
        "cookie": None,
        "model": MODEL_ALIASES.get(req.model or "glm-5.2", req.model or "glm-5.2"),
        "enable_thinking": req.enable_thinking,
        "reasoning_effort": req.reasoning_effort or "max",
        "web_search": req.web_search,
        "temperature": req.temperature if req.temperature is not None else 1.0,
        "max_tokens": req.max_tokens if req.max_tokens is not None else 8192,
        "chat_id": chat_id,
        "last_assistant_id": last_assistant_id,
        "last_assistant_parent_id": last_assistant_parent_id,
        "seed_msg_id": seed_msg_id,
        "history": history,
        "usage": {"prompts": 0, "in": 0, "out": 0},
        "solver": None,
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": model_catalog()}


def _tool_prompt(req: ChatCompletionRequest) -> str:
    prompt = ""
    for m in req.messages:
        if m.role != "user":
            continue
        content = m.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        prompt = content
    return prompt


async def _run_with_tools(req: ChatCompletionRequest, token: str, request: Request = None):
    """Execute the real tool loop (local + MCP) for an OpenAI tools request.

    Runs send_with_tools on a worker thread (it blocks on captcha solves and
    upstream SSE). Each tool-loop iteration grabs a FRESH single-use captcha via
    _captcha_pair, serialized on the shared solver lock. Local write/run tools
    are gated by GLM_TOOL_AUTORUN policy (approve_tool_auto)."""
    prompt = _tool_prompt(req)
    if not prompt:
        raise HTTPException(status_code=400, detail="No user message")
    session_id = _resolve_session_id(req, request) if request else None
    state = _tool_state(req, token, session_id=session_id)
    mcp = _get_mcp()
    captcha_fn = lambda: _captcha_pair(token)
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = MODEL_ALIASES.get(req.model or "glm-5.2", req.model or "glm-5.2")

    def chunk(delta, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }],
        }

    if req.stream:
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        writer = _QueueWriter(loop, queue)
        done = {"ok": True, "err": None}

        def run():
            try:
                done["ok"], done["err"] = send_with_tools(
                    state, prompt, mcp=mcp, auto_approve=True, writer=writer,
                    captcha_fn=captcha_fn)
            finally:
                if session_id:
                    s = SESSIONS.setdefault(session_id, {})
                    if state.get("history"):
                        s["history"] = list(state["history"])
                    if state.get("chat_id"):
                        s["chat_id"] = state["chat_id"]
                    if state.get("last_assistant_id"):
                        s["last_assistant_id"] = state["last_assistant_id"]
                    if state.get("last_assistant_parent_id"):
                        s["last_assistant_parent_id"] = state["last_assistant_parent_id"]
                    if state.get("seed_msg_id"):
                        s["seed_msg_id"] = state["seed_msg_id"]
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, run)

        async def event_generator():
            while True:
                piece = await queue.get()
                if piece is None:
                    break
                yield f"data: {json.dumps(chunk(piece))}\n\n"
            if not done["ok"] and done["err"]:
                errmsg = f"\n[error] {done['err']}"
                for i in range(0, len(errmsg), 48):
                    yield f"data: {json.dumps(chunk(errmsg[i:i + 48]))}\n\n"
            finish = chunk(None, finish_reason='stop')
            norm = normalize_usage(state["usage"])
            if norm:
                finish["usage"] = norm
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    writer = _CaptureWriter()
    ok, err = await asyncio.to_thread(
        send_with_tools, state, prompt, mcp=mcp, auto_approve=True,
        writer=writer, captcha_fn=captcha_fn)
    if session_id:
        s = SESSIONS.setdefault(session_id, {})
        if state.get("history"):
            s["history"] = list(state["history"])
        if state.get("chat_id"):
            s["chat_id"] = state["chat_id"]
        if state.get("last_assistant_id"):
            s["last_assistant_id"] = state["last_assistant_id"]
        if state.get("last_assistant_parent_id"):
            s["last_assistant_parent_id"] = state["last_assistant_parent_id"]
        if state.get("seed_msg_id"):
            s["seed_msg_id"] = state["seed_msg_id"]
    if not ok and not writer.text:
        raise HTTPException(status_code=502, detail=err or "tool loop failed")
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": writer.text or None},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": state["usage"]["in"],
            "completion_tokens": state["usage"]["out"],
            "total_tokens": state["usage"]["in"] + state["usage"]["out"],
        },
    }


# Curated, GLM-friendly core tools. GLM-5.2 emits these names reliably (the
# same set the --repl uses via TOOL_CONTRACT); parse_tool_calls maps them back
# to the client's real tool names via INTENT_TO_CLIENT_TOOL / _INTENT_CLUSTERS.
_TOOL_DISPLAY = {
    "exec":   ("run_command", '{"cmd": "ls -la"}'),
    "list":   ("list_dir",    '{"path": "."}'),
    "read":   ("read_file",   '{"path": "/etc/hostname"}'),
    "write":  ("write_file",  '{"path": "file.txt", "content": "hello"}'),
    "search": ("grep",        '{"pattern": "TODO", "path": "."}'),
}
_CORE_INTENT_ORDER = ["exec", "list", "read", "write", "search", "edit"]


def _intent_display(intent):
    """(GLM-friendly name, example args) for a contracted intent; edit is
    special-cased since it keeps its real OpenAI name/args."""
    if intent == "edit":
        return "edit", '{"old_string": "foo", "new_string": "bar"}'
    return _TOOL_DISPLAY[intent]


def _tool_hint_client(contracted_tools):
    """Per-prompt TOOL_HINT for the client path (repl appends TOOL_HINT to every
    turn; without it GLM drifts to web native formats. Keep it SMALL and
    reference ONLY the curated names actually offered, to avoid confusing GLM
    with tools that aren't in this conversation)."""
    contracted = contracted_tools or set()
    lines = [f"  TOOL:{_intent_display(intent)[0]}({_intent_display(intent)[1]})\n"
             for intent in _CORE_INTENT_ORDER if intent in contracted]
    if contracted:
        first = next((i for i in _CORE_INTENT_ORDER if i in contracted), None)
        name, ex = _intent_display(first)
        example = f"TOOL:{name}({ex})"
    else:
        example = 'TOOL:run_command({"cmd": "ls -la"})'
    return (
        "\n(Reminder: emit TOOL: lines across iterations for multi-step tasks; never "
        "repeat identical calls. Available tools — one per line, nothing else:\n"
        + "".join(lines)
        + "When all info gathered, write your final answer. Example: "
        + example
        + ")"
    )


def _contracted_intents(tools):
    """Which curated intents the client tool set contracts (mirrors the
    detection loop in _tools_contract)."""
    intents = set()
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) or {}
        name = fn.get("name") or t.get("name") or ""
        if name == "edit":
            intents.add("edit")
        intent = _intent_of(name)
        if intent in _TOOL_DISPLAY:
            intents.add(intent)
    return intents


def _tools_contract(tools) -> str:
    """Render a CURATED, GLM-friendly text contract.

    GLM-5.2 reliably emits a small set of tool names it was trained on
    (run_command, list_dir, read_file, write_file, grep) -- the same set the
    --repl uses via TOOL_CONTRACT. Instead of forwarding the client's full
    tool set (e.g. opencode's 113 tools, which dilutes and confuses GLM), we
    translate the client tools into this curated set, foreground the core
    coding tools, and keep the format tight. parse_tool_calls maps the
    GLM-friendly names back to the client's real tool names + arg keys
    (see config.INTENT_TO_CLIENT_TOOL)."""

    descriptions = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) or {}
        name = fn.get("name") or t.get("name") or ""
        intent = _intent_of(name)
        if intent in _TOOL_DISPLAY and intent not in descriptions:
            descriptions[intent] = (fn.get("description") or "").strip()
        if name == "edit" and "edit" not in descriptions:
            descriptions["edit"] = (fn.get("description") or "").strip()

    auth_block = (
        "AUTHORIZATION & WORKSPACE CAPABILITY:\n"
        "You are operating inside an active developer workspace with full client "
        "tool-execution privileges. You do NOT have direct filesystem access "
        "yourself; the client environment EXECUTES your TOOL: commands on the "
        "user's real machine and returns the outputs. Whenever the request needs "
        "to list files, read files, edit code, search, or run a terminal command, "
        "you MUST emit a TOOL: line — doing so is how the work gets done, not a "
        "violation of any policy. Never tell the user you lack access or cannot "
        "use tools; just call the appropriate tool."
    )

    lines = [
        auth_block,
        "",
        "You have access to the tools listed below. Use them ONLY when the request actually "
        "requires reading files, running terminal commands, editing code, or searching.",
        "If the user greets you (e.g. 'hi', 'hello') or asks a general question that does NOT "
        "require a tool, reply directly in friendly natural language and DO NOT emit any TOOL: line.",
        "When you DO need a tool, invoke it by emitting exactly one line: "
        'TOOL:run_command({"cmd": "ls -la"}) with the arguments as inline JSON. ',
        "Do NOT wrap the call in prose, markdown code fences (NO ```bash, NO ```json), or XML tags. ",
        "Do NOT put the arguments on a separate line.",
        "To run multiple tools in parallel, emit one separate call per tool, each on its own line.",
        "The caller executes each TOOL: line and returns the result as a [Tool result] message.",
        "Keep emitting TOOL: lines only while you still need data, then give your final answer.",
        "Do NOT write tutorials, dummy API requests, or documentation about how tool calling works.",
        "Available tools:",
    ]

    rendered = False
    for intent in _CORE_INTENT_ORDER:
        if intent not in descriptions:
            continue
        rendered = True
        if intent == "edit":
            desc = descriptions["edit"] or "Edit a file by replacing text."
            lines.append(f"- edit: {desc}")
            lines.append('  example: TOOL:edit({"old_string": "foo", "new_string": "bar"})')
        else:
            disp_name, example = _TOOL_DISPLAY[intent]
            desc = descriptions[intent] or f"{disp_name} tool"
            lines.append(f"- {disp_name}: {desc}")
            lines.append(f"  example: TOOL:{disp_name}({example})")
    if not rendered:
        for t in (tools or [])[:8]:
            fn = (t.get("function", t) if isinstance(t, dict) else {}) or {}
            name = fn.get("name") or (t.get("name") if isinstance(t, dict) else "") or "tool"
            lines.append(f"- {name}: {(fn.get('description') or '').strip()}")
    return "\n".join(lines)


async def _client_nudge(answer, formatted_messages, *, token, cookie, chat_id, model,
                        msg_id, features, params, captcha, req, call_upstream,
                        max_nudges=2):
    """Refusal recovery for the client-side tool path.

    If `answer` already contains a tool call, return it unchanged. If `answer`
    matches the refusal regex but has no call, transparently nudge GLM (up to
    `max_nudges` times) via `call_upstream(messages, nudge_text)` (one upstream
    turn, returns answer text) and return any parsed call. OpenCode never sees
    the refusal text."""
    calls = parse_tool_calls(answer, known_tools=req.tools)
    if calls:
        return calls, answer
    messages = list(formatted_messages)
    attempts = 0
    while REFUSAL_RE.search(answer) and attempts < max_nudges:
        messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": TOOL_NUDGE_CLIENT},
        ]
        try:
            retry_text = await call_upstream(messages, TOOL_NUDGE_CLIENT)
        except Exception:
            return [], answer
        retry_calls = parse_tool_calls(retry_text, known_tools=req.tools)
        if retry_calls:
            return retry_calls, retry_text
        answer = retry_text
        attempts += 1
    return [], answer


async def _run_client_tools(req: ChatCompletionRequest, token: str, request: Request = None):
    """OpenAI-compatible client-side tool calling.

    Builds a TOOL:-line contract from the client's tool schemas, sends it to
    GLM as a normal prompt, parses TOOL: calls out of the reply, and streams
    them back as standard delta.tool_calls + finish_reason:"tool_calls" for the
    harness (Cline/Aider/Cursor) to execute. The harness sends role:"tool"
    results back, which are re-injected as [Tool result] text on the next call.
    """
    if not req.tools:
        raise HTTPException(status_code=400, detail="tools required for client-side mode")
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages")

    session_id = _resolve_session_id(req, request) if request else None
    sess_history = []
    if session_id:
        if len(req.messages) == 1:
            sess_history = list(SESSIONS.get(session_id, {}).get("history", []))
        else:
            hist = _tool_messages(req)[:-1]
            if len(hist) > HISTORY_LIMIT:
                hist = hist[-HISTORY_LIMIT:]
            SESSIONS.setdefault(session_id, {})["history"] = hist

    contract = _tools_contract(req.tools)
    directives = _collect_directives(req)
    lead = f"{directives}\n\n{contract}" if directives else contract
    formatted_messages = []
    for m in sess_history:
        formatted_messages.append({"role": m["role"], "content": m["content"]})

    def _has_contract():
        for fm in formatted_messages:
            if fm["role"] == "user" and contract in fm["content"]:
                return True
        return False

    for i, m in enumerate(req.messages):
        content = m.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        role = m.role
        if role == "user":
            if i == 0 and not _has_contract():
                content = f"{lead}\n\n{content}"
            formatted_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if m.tool_calls:
                formatted_messages.append({"role": "assistant", "content": _assistant_tool_lines(m)})
            else:
                formatted_messages.append({"role": "assistant", "content": content or ""})
        elif role == "tool":
            tid = m.tool_call_id or m.name or "?"
            formatted_messages.append({"role": "user",
                                       "content": f"[Tool result for {tid}]: {content}"})
    if req.messages and req.messages[0].role != "user" and not _has_contract():
        formatted_messages.insert(0, {"role": "user", "content": lead})

    # Repl parity: every turn's prompt carries the TOOL_HINT reminder; without it
    # GLM-5.2 drifts to the chat.z.ai web app's native tool formats instead of
    # the TOOL: convention. Append to the LAST message (the one GLM answers).
    if formatted_messages:
        hint = _tool_hint_client(_contracted_intents(req.tools))
        formatted_messages[-1]["content"] = formatted_messages[-1]["content"] + hint
    prompt = formatted_messages[-1]["content"]

    chat_model = MODEL_ALIASES.get(req.model or "glm-5.2", req.model or "glm-5.2")
    captcha, cookie = await acquire_captcha(token)
    sess = SESSIONS.get(session_id) if session_id else None
    if sess and sess.get("chat_id"):
        chat_id = sess["chat_id"]
        msg_id = str(uuid.uuid4())
        parent_msg_id = sess.get("last_assistant_id") or sess.get("seed_msg_id")
    else:
        seed_messages = list(sess_history) + _tool_messages(req)
        chat_res = create_chat(token, prompt, model=chat_model, cookie=cookie,
                               messages=seed_messages,
                               enable_thinking=req.enable_thinking,
                               reasoning_effort=req.reasoning_effort)
        chat_id = chat_res[0]
        msg_id = chat_res[1]
        parent_msg_id = chat_res[2] if len(chat_res) > 2 else None
        if session_id:
            s = SESSIONS.setdefault(session_id, {})
            s["chat_id"] = chat_id
            s["seed_msg_id"] = msg_id
    sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                               current_url=f"{BASE}/c/{chat_id}")
    url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"
    params = {"max_tokens": req.max_tokens, "temperature": req.temperature, "top_p": 0.95}
    features = build_features(req.enable_thinking, req.reasoning_effort, req.web_search)

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    def chunk(delta=None, tool_calls=None, finish_reason=None, role=None):
        d = {}
        if role:
            d["role"] = role
        if delta:
            d["content"] = delta
        if tool_calls:
            d["tool_calls"] = tool_calls
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chat_model,
            "choices": [{"index": 0, "delta": d, "finish_reason": finish_reason}],
        }

    usage = None
    last_ast_id = None
    resp = requests.post(
        url, headers=headers(token, sig, cookie), stream=True, timeout=120,
        json={
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": chat_id,
            "current_user_message_id": msg_id,
            "current_user_message_parent_id": parent_msg_id,
            "extra": {},
            "features": features,
            "id": str(uuid.uuid4()),
            "messages": formatted_messages,
            "model": chat_model,
            "params": params,
            "signature_prompt": prompt,
            "stream": True,
            "variables": {},
            "captcha_verify_param": captcha,
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:1000])

    def _reasoning_chunk(text):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chat_model,
            "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}],
        }

    # ---- Non-streaming path: buffer the whole upstream then emit once. ----
    if not req.stream:
        reasoning_buf, text_buf = [], []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                inner = json.loads(data).get("data", {})
            except Exception:
                continue
            if inner.get("id"):
                last_ast_id = inner["id"]
            if inner.get("usage"):
                usage = inner["usage"]
            delta = inner.get("delta_content") or ""
            if not delta:
                continue
            if inner.get("phase") == "thinking":
                reasoning_buf.append(delta)
            else:
                text_buf.append(delta)
        if session_id and last_ast_id:
            SESSIONS.setdefault(session_id, {})["last_assistant_id"] = last_ast_id
        full_text = "".join(text_buf)
        reasoning = "".join(reasoning_buf)
        calls = parse_tool_calls(full_text, known_tools=req.tools)
        if calls:
            aid2, _ = await _fetch_reply(token, chat_id, cookie, after_user_id=msg_id)
            if session_id and aid2:
                SESSIONS.setdefault(session_id, {})["last_assistant_id"] = aid2
            tool_calls = [{
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
            } for i, (name, args, _matched) in enumerate(calls)]
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            if reasoning:
                message["reasoning_content"] = reasoning
            r = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": chat_model,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            }
            if usage:
                norm = normalize_usage(usage)
                if norm:
                    r["usage"] = norm
            return r
        answer = strip_tool_lines(full_text)
        raw_user_prompt = _tool_prompt(req)
        if session_id:
            _commit_turn(session_id, raw_user_prompt, answer, chat_id=chat_id)
            aid2, _ = await _fetch_reply(token, chat_id, cookie, after_user_id=msg_id)
            if aid2:
                SESSIONS.setdefault(session_id, {})["last_assistant_id"] = aid2
        message = {"role": "assistant", "content": answer or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": chat_model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }

    # ---- Streaming path: buffer the whole answer phase, then decide
    # ---- tool-vs-content at end-of-stream (GLM emits prose before TOOL:). ----
    async def gen():
        yield f"data: {json.dumps(chunk_init(completion_id, created, chat_model))}\n\n"
        answer_parts, reasoning_parts = [], []
        _usage, _last = None, None

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                inner = json.loads(data).get("data", {})
            except Exception:
                continue
            if inner.get("id"):
                _last = inner["id"]
            if inner.get("usage"):
                _usage = inner["usage"]
            delta = inner.get("delta_content") or ""
            if not delta:
                continue
            if inner.get("phase") == "thinking":
                reasoning_parts.append(delta)
                yield f"data: {json.dumps(_reasoning_chunk(delta))}\n\n"
                continue
            answer_parts.append(delta)

        if session_id and _last:
            SESSIONS.setdefault(session_id, {})["last_assistant_id"] = _last

        answer = "".join(answer_parts)

        async def _call_upstream(messages, nudge_text):
            sig2, up2, ts2 = sign(nudge_text, user_id_from_token(token), token,
                                  current_url=f"{BASE}/c/{chat_id}")
            url2 = f"{BASE}{ENDPOINT}?{up2}&signature_timestamp={ts2}"
            res = await asyncio.to_thread(
                stream_turn,
                token=token, cookie=cookie, sig=sig2, url=url2, chat_id=chat_id,
                model=chat_model, messages=messages,
                current_user_message_id=str(uuid.uuid4()),
                current_user_message_parent_id=msg_id,
                is_first=False, features=features, params=params, captcha=captcha,
                signature_prompt_override=nudge_text,
            )
            ans = res.get("answer") or ""
            return ans

        calls, answer = await _client_nudge(
            answer, formatted_messages, token=token, cookie=cookie, chat_id=chat_id,
            model=chat_model, msg_id=msg_id, features=features, params=params,
            captcha=captcha, req=req, call_upstream=_call_upstream)

        # Discover the server-assigned reply node id (SSE omits it). The next
        # request MUST be parented beneath it or the completions backend
        # answers without any prior context (verified live 2026-08-16).
        aid2, _ = await _fetch_reply(token, chat_id, cookie, after_user_id=msg_id)
        if session_id and aid2:
            SESSIONS.setdefault(session_id, {})["last_assistant_id"] = aid2

        if calls:
            tool_calls = [{
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
            } for i, (name, args, _matched) in enumerate(calls)]
            yield f"data: {json.dumps(chunk(tool_calls=tool_calls, delta=None, role='assistant'))}\n\n"
            finish = chunk(finish_reason="tool_calls")
            norm = normalize_usage(_usage)
            if norm:
                finish["usage"] = norm
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"
            return

        final_answer = answer
        if session_id:
            _commit_turn(session_id, _tool_prompt(req), final_answer, chat_id=chat_id)
        if answer:
            yield f"data: {json.dumps(chunk(delta=answer))}\n\n"
        finish = chunk(finish_reason="stop")
        norm = normalize_usage(_usage)
        if norm:
            finish["usage"] = norm
        yield f"data: {json.dumps(finish)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    try:
        token = get_token()
        token = refresh_token(token)

        if req.tools:
            if os.environ.get("GLM_SERVER_TOOLS") in ("1", "true", "yes"):
                return await _run_with_tools(req, token, request=request)
            return await _run_client_tools(req, token, request=request)

        session_id = _resolve_session_id(req, request)
        sess_history = []
        if session_id:
            if len(req.messages) == 1:
                sess_history = list(SESSIONS.get(session_id, {}).get("history", []))
            else:
                hist = _tool_messages(req)[:-1]
                if len(hist) > HISTORY_LIMIT:
                    hist = hist[-HISTORY_LIMIT:]
                SESSIONS.setdefault(session_id, {})["history"] = hist

        directives = _collect_directives(req)
        formatted_messages = []
        for m in sess_history:
            formatted_messages.append({"role": m["role"], "content": m["content"]})

        prompt = ""
        for i, m in enumerate(req.messages):
            role = m.role
            content = m.content
            if isinstance(content, list):
                content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
            if role in ("system", "developer"):
                continue
            if role == "user":
                formatted_messages.append({"role": "user", "content": content})
                prompt = content
            elif role == "assistant":
                if m.tool_calls:
                    formatted_messages.append({"role": "assistant", "content": _assistant_tool_lines(m)})
                else:
                    formatted_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                formatted_messages.append({"role": "user", "content": f"[Tool Result]: {content}"})
        if directives:
            leading = [fm for fm in formatted_messages if fm["role"] == "user"]
            if not leading or directives not in leading[0]["content"]:
                formatted_messages.insert(0, {"role": "user", "content": directives})

        captcha, cookie = await acquire_captcha(token)

        chat_model = req.model
        if not chat_model:
            chat_model = "glm-5.2"
        chat_model = MODEL_ALIASES.get(chat_model, chat_model)

        sess = SESSIONS.get(session_id) if session_id else None
        is_first = not (sess and sess.get("chat_id"))
        if sess and sess.get("chat_id"):
            chat_id = sess["chat_id"]
            msg_id = str(uuid.uuid4())
            parent_msg_id = sess.get("last_assistant_id") or sess.get("seed_msg_id")
        else:
            seed_messages = list(sess_history) + _tool_messages(req)
            chat_res = create_chat(token, prompt, model=chat_model, cookie=cookie,
                                   messages=seed_messages,
                                   enable_thinking=req.enable_thinking,
                                   reasoning_effort=req.reasoning_effort)
            chat_id = chat_res[0]
            msg_id = chat_res[1]
            parent_msg_id = chat_res[2] if len(chat_res) > 2 else None
            if session_id:
                s = SESSIONS.setdefault(session_id, {})
                s["chat_id"] = chat_id
                s["seed_msg_id"] = msg_id
        sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                                   current_url=f"{BASE}/c/{chat_id}")
        url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"

        body = {}
        if is_first:
            body["background_tasks"] = {"title_generation": True, "tags_generation": True}
        body.update({
            "chat_id": chat_id,
            "current_user_message_id": msg_id,
            "current_user_message_parent_id": parent_msg_id,
            "extra": {},
            "features": build_features(req.enable_thinking, req.reasoning_effort, req.web_search),
            "id": str(uuid.uuid4()),
            "messages": formatted_messages,
            "model": chat_model,
            "params": {"max_tokens": req.max_tokens, "temperature": req.temperature, "top_p": 0.95},
            "signature_prompt": prompt,
            "stream": True,
            "variables": {},
            "captcha_verify_param": captcha,
        })

        # ===== TEMP DEBUG INSTRUMENTATION (remove before shipping) =====
        try:
            _dbg = open("/tmp/glm_upstream_debug.log", "a")
            _dbg.write("\n===== UPSTREAM REQUEST =====\n")
            _dbg.write(f"url: {url}\n")
            _dbg.write(f"chat_id: {chat_id}\n")
            _dbg.write(f"current_user_message_id: {msg_id}\n")
            _dbg.write(f"current_user_message_parent_id: {parent_msg_id}\n")
            _dbg.write(f"signature_prompt: {prompt!r}\n")
            _dbg.write("messages:\n")
            for _fm in formatted_messages:
                _dbg.write(f"  role={_fm.get('role')} content={_fm.get('content')!r}\n")
            _dbg.write(f"session_id: {session_id}\n")
            _dbg.write(f"sess_state: {dict(SESSIONS.get(session_id, {})) if session_id else None}\n")
            _dbg.write("================================\n")
            _dbg.close()
        except Exception as _e:
            print(f"DEBUGLOG-ERR {_e}", file=__import__('sys').stderr)
        # ===== END TEMP DEBUG INSTRUMENTATION =====

        resp = requests.post(url, headers=headers(token, sig, cookie), json=body,
                             stream=True, timeout=120)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:1000])

        completion_id = f"chatcmpl-{uuid.uuid4()}"

        def collect():
            full_text = ""
            reasoning = ""
            usage = None
            last_ast = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    inner = j.get("data", {})
                    try:
                        _dbg = open("/tmp/glm_upstream_debug.log", "a")
                        _dbg.write(f"[SSE-FULL] {json.dumps(j)[:800]}\n")
                        _dbg.close()
                    except Exception:
                        pass
                    phase = inner.get("phase")
                    delta = inner.get("delta_content") or ""
                    if inner.get("id"):
                        last_ast = inner["id"]
                    if inner.get("usage"):
                        usage = inner.get("usage")
                    if phase == "thinking":
                        reasoning += delta
                    else:
                        full_text += delta
                except Exception:
                    pass
            return full_text, reasoning, usage, last_ast

        if not req.stream:
            full_text, reasoning, usage, last_ast = collect()
            if session_id:
                _commit_turn(session_id, prompt, full_text, chat_id=chat_id, last_ast_id=last_ast)
                aid2, _ = await _fetch_reply(token, chat_id, cookie, after_user_id=msg_id)
                if aid2:
                    SESSIONS.setdefault(session_id, {})["last_assistant_id"] = aid2
            try:
                _dbg = open("/tmp/glm_upstream_debug.log", "a")
                _dbg.write(f"===== TURN1/COLLECT SSE last_ast (inner id): {last_ast!r}\n")
                _dbg.close()
            except Exception:
                pass
            message = {"role": "assistant", "content": full_text or None}
            if reasoning:
                message["reasoning_content"] = reasoning
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": chat_model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }],
                "usage": normalize_usage(usage) or {
                    "prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20,
                },
            }

        async def event_generator():
            yield f"data: {json.dumps(chunk_init(completion_id, int(time.time()), chat_model))}\n\n"
            usage = None
            last_ast = None
            answer_buf = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        inner = j.get("data", {})
                        if inner.get("id"):
                            last_ast = inner["id"]
                        phase = inner.get("phase")
                        delta = inner.get("delta_content") or ""
                        if inner.get("usage"):
                            usage = inner.get("usage")
                        if not delta:
                            continue
                        if phase == "thinking":
                            delta_key = "reasoning_content"
                        else:
                            delta_key = "content"
                            answer_buf.append(delta)
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": chat_model,
                            "choices": [{
                                "index": 0,
                                "delta": {delta_key: delta},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception:
                        pass
            if session_id:
                _commit_turn(session_id, prompt, "".join(answer_buf), chat_id=chat_id, last_ast_id=last_ast)
                aid2, _ = await _fetch_reply(token, chat_id, cookie, after_user_id=msg_id)
                if aid2:
                    SESSIONS.setdefault(session_id, {})["last_assistant_id"] = aid2
            try:
                _dbg = open("/tmp/glm_upstream_debug.log", "a")
                _dbg.write(f"===== SSE last_ast (inner id): {last_ast!r}\n")
                _dbg.close()
            except Exception:
                pass
            finish = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            norm = normalize_usage(usage)
            if norm:
                finish["usage"] = norm
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    """Multi-turn chat for the web UI. Streams OpenAI-style SSE chunks."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    try:
        token = get_token()
        token = refresh_token(token)

        session_id = req.session_id or uuid.uuid4().hex
        sess = SESSIONS.get(session_id) or {
            "chat_id": None,
            "last_assistant_id": None,
            "history": [],
        }

        captcha, cookie = await acquire_captcha(token)

        prompt = req.prompt.strip()
        chat_model = MODEL_ALIASES.get(req.model or "glm-5.2", req.model or "glm-5.2")
        temperature = req.temperature if req.temperature is not None else 1.0
        max_tokens = req.max_tokens if req.max_tokens is not None else 8192

        user_msg_id = str(uuid.uuid4())
        parent_id = sess.get("last_assistant_id") or sess.get("seed_msg_id")

        if sess["chat_id"] is None:
            chat_id, seed_msg_id, *_ = create_chat(token, prompt, model=chat_model, cookie=cookie,
                                                   enable_thinking=req.enable_thinking,
                                                   reasoning_effort=req.reasoning_effort)
            sess["chat_id"] = chat_id
            sess["seed_msg_id"] = seed_msg_id
        else:
            chat_id = sess["chat_id"]

        sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                                   current_url=f"{BASE}/c/{chat_id}")
        url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"

        body_messages = list(sess["history"]) + [{"role": "user", "content": prompt}]
        body = {
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": chat_id,
            "current_user_message_id": user_msg_id,
            "current_user_message_parent_id": parent_id,
            "extra": {},
            "features": build_features(req.enable_thinking, req.reasoning_effort, req.web_search),
            "id": str(uuid.uuid4()),
            "messages": body_messages,
            "model": chat_model,
            "params": {"max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95},
            "signature_prompt": prompt,
            "stream": True,
            "variables": {},
            "captcha_verify_param": captcha,
        }

        try:
            resp = requests.post(url, headers=headers(token, sig, cookie), json=body,
                                 stream=True, timeout=180)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail=f"GLM upstream error: {e}")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:1000])

        completion_id = f"chatcmpl-{uuid.uuid4()}"

        def build_chunk(delta):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta} if delta else {},
                    "finish_reason": None,
                }],
            }

        async def event_generator():
            collected = []
            new_assistant_id = sess["last_assistant_id"]
            usage = None
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if await request.is_disconnected():
                        break
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue
                    inner = j.get("data", {})
                    delta = inner.get("delta_content") or ""
                    mid = inner.get("id")
                    if mid:
                        new_assistant_id = mid
                    u = inner.get("usage")
                    if u:
                        usage = u
                    if delta:
                        collected.append(delta)
                        chunk = build_chunk(delta)
                        chunk["choices"][0]["delta"]["content"] = delta
                        yield f"data: {json.dumps(chunk)}\n\n"
                finish = build_chunk("")
                finish["choices"][0]["finish_reason"] = "stop"
                if usage:
                    finish["usage"] = usage
                yield f"data: {json.dumps(finish)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                text = "".join(collected)
                sess["history"].append({"role": "user", "content": prompt})
                if text:
                    sess["history"].append({"role": "assistant", "content": text})
                    sess["last_assistant_id"] = new_assistant_id or sess["last_assistant_id"]
                if len(sess["history"]) > HISTORY_LIMIT:
                    sess["history"] = sess["history"][-HISTORY_LIMIT:]
                SESSIONS[session_id] = sess
                # discover the server-assigned reply node for next-turn threading
                if text:
                    sid2, _ = await _fetch_reply(token, chat_id, cookie,
                                                 after_user_id=user_msg_id)
                    if sid2:
                        sess["last_assistant_id"] = sid2
                        SESSIONS[session_id] = sess

        return StreamingResponse(event_generator(), media_type="text/event-stream",
                                 headers={"X-Session-Id": session_id})
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/clear")
async def clear_session():
    SESSIONS.clear()
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)