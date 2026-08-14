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
from glm_rev.client import create_chat, refresh_token, build_features
from glm_rev.config import BASE, ENDPOINT, parse_tool_calls, strip_tool_lines
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


def _tool_state(req: ChatCompletionRequest, token: str) -> dict:
    return {
        "token": token,
        "cookie": None,
        "model": MODEL_ALIASES.get(req.model or "glm-5.2", req.model or "glm-5.2"),
        "enable_thinking": req.enable_thinking,
        "reasoning_effort": req.reasoning_effort or "max",
        "web_search": req.web_search,
        "temperature": req.temperature if req.temperature is not None else 1.0,
        "max_tokens": req.max_tokens if req.max_tokens is not None else 8192,
        "chat_id": None,
        "last_assistant_id": None,
        "last_assistant_parent_id": None,
        "history": [],
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


async def _run_with_tools(req: ChatCompletionRequest, token: str):
    """Execute the real tool loop (local + MCP) for an OpenAI tools request.

    Runs send_with_tools on a worker thread (it blocks on captcha solves and
    upstream SSE). Each tool-loop iteration grabs a FRESH single-use captcha via
    _captcha_pair, serialized on the shared solver lock. Local write/run tools
    are gated by GLM_TOOL_AUTORUN policy (approve_tool_auto)."""
    prompt = _tool_prompt(req)
    if not prompt:
        raise HTTPException(status_code=400, detail="No user message")
    state = _tool_state(req, token)
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


def _tools_contract(tools) -> str:
    """Render the client's tool schemas into a TOOL:-line text contract."""
    lines = [
        "You are a coding agent. To invoke a tool, emit exactly one line:",
        "  TOOL:<name>({\"arg\": \"value\", ...})",
        "Then the caller executes it and returns the result as a [Tool result] message.",
        "Continue emitting TOOL: lines until you have all data, then give your final answer.",
        "Available tools:",
    ]
    for i, t in enumerate(tools or []):
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) or {}
        name = fn.get("name") or t.get("name") or f"tool_{i}"
        desc = (fn.get("description") or t.get("description") or "").strip()
        params = fn.get("parameters") or t.get("parameters") or {}
        lines.append(f"- {name}: {desc}")
        if params:
            lines.append(f"  args schema: {json.dumps(params)}")
    return "\n".join(lines)


async def _run_client_tools(req: ChatCompletionRequest, token: str):
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

    contract = _tools_contract(req.tools)
    formatted_messages = []
    for i, m in enumerate(req.messages):
        content = m.content
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        role = m.role
        if role == "user":
            if i == len(req.messages) - 1:
                content = f"{contract}\n\n{content}"
            formatted_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            formatted_messages.append({"role": "assistant", "content": content or ""})
        elif role == "tool":
            tid = m.tool_call_id or m.name or "?"
            formatted_messages.append({"role": "user",
                                       "content": f"[Tool result for {tid}]: {content}"})
    prompt = formatted_messages[-1]["content"]

    chat_model = MODEL_ALIASES.get(req.model or "glm-5.2", req.model or "glm-5.2")
    captcha, cookie = await acquire_captcha(token)
    chat_id, msg_id = create_chat(token, prompt, model=chat_model, cookie=cookie,
                                  enable_thinking=req.enable_thinking,
                                  reasoning_effort=req.reasoning_effort)
    sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                               current_url=f"{BASE}/c/{chat_id}")
    url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"
    params = {"max_tokens": req.max_tokens, "temperature": req.temperature, "top_p": 0.95}

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

    reasoning_buf, text_buf = [], []
    for line in requests.post(
        url, headers=headers(token, sig, cookie), stream=True, timeout=120,
        json={
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": chat_id,
            "current_user_message_id": msg_id,
            "current_user_message_parent_id": None,
            "extra": {},
            "features": build_features(req.enable_thinking, req.reasoning_effort, req.web_search),
            "id": str(uuid.uuid4()),
            "messages": formatted_messages,
            "model": chat_model,
            "params": params,
            "signature_prompt": prompt,
            "stream": True,
            "variables": {},
            "captcha_verify_param": captcha,
        },
    ).iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            inner = json.loads(data).get("data", {})
        except Exception:
            continue
        delta = inner.get("delta_content") or ""
        if not delta:
            continue
        if inner.get("phase") == "thinking":
            reasoning_buf.append(delta)
        else:
            text_buf.append(delta)

    full_text = "".join(text_buf)
    reasoning = "".join(reasoning_buf)
    calls = parse_tool_calls(full_text)

    if calls:
        tool_calls = [{
            "index": i,
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
        } for i, (name, args, _matched) in enumerate(calls)]

        if req.stream:
            async def gen():
                yield f"data: {json.dumps(chunk(tool_calls=tool_calls, delta=None, role='assistant'))}\n\n"
                yield f"data: {json.dumps(chunk(finish_reason='tool_calls'))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": chat_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }],
        }

    answer = strip_tool_lines(full_text)
    message = {"role": "assistant", "content": answer or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if req.stream:
        async def gen():
            yield f"data: {json.dumps(chunk_init(completion_id, created, chat_model))}\n\n"
            if reasoning:
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': chat_model, 'choices': [{'index': 0, 'delta': {'reasoning_content': reasoning}, 'finish_reason': None}]})}\n\n"
            if answer:
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': chat_model, 'choices': [{'index': 0, 'delta': {'content': answer}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps(chunk(finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": chat_model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    try:
        token = get_token()
        token = refresh_token(token)

        if req.tools:
            if os.environ.get("GLM_CLIENT_TOOLS") in ("1", "true", "yes"):
                return await _run_client_tools(req, token)
            return await _run_with_tools(req, token)

        formatted_messages = []
        prompt = ""
        for i, m in enumerate(req.messages):
            role = m.role
            content = m.content
            if isinstance(content, list):
                content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
            if role == "user":
                formatted_messages.append({"role": "user", "content": content})
                prompt = content
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                formatted_messages.append({"role": "user", "content": f"[Tool Result]: {content}"})

        captcha, cookie = await acquire_captcha(token)

        chat_model = req.model
        if not chat_model:
            chat_model = "glm-5.2"
        chat_model = MODEL_ALIASES.get(chat_model, chat_model)

        chat_id, msg_id = create_chat(token, prompt, model=chat_model, cookie=cookie,
                                      enable_thinking=req.enable_thinking,
                                      reasoning_effort=req.reasoning_effort)
        sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                                   current_url=f"{BASE}/c/{chat_id}")
        url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"

        body = {
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": chat_id,
            "current_user_message_id": msg_id,
            "current_user_message_parent_id": None,
            "extra": {},
            "features": build_features(req.enable_thinking, req.reasoning_effort, req.web_search),
            "id": str(uuid.uuid4()),
            "messages": formatted_messages,
            "model": chat_model,
            "params": {"max_tokens": req.max_tokens, "temperature": req.temperature, "top_p": 0.95},
            "signature_prompt": prompt,
            "stream": req.stream,
            "variables": {},
            "captcha_verify_param": captcha,
        }

        resp = requests.post(url, headers=headers(token, sig, cookie), json=body,
                             stream=True, timeout=120)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:1000])

        completion_id = f"chatcmpl-{uuid.uuid4()}"

        def collect():
            full_text = ""
            reasoning = ""
            usage = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    inner = j.get("data", {})
                    phase = inner.get("phase")
                    delta = inner.get("delta_content") or ""
                    if inner.get("usage"):
                        usage = inner.get("usage")
                    if phase == "thinking":
                        reasoning += delta
                    else:
                        full_text += delta
                except Exception:
                    pass
            return full_text, reasoning, usage

        if not req.stream:
            full_text, reasoning, usage = collect()
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
        parent_id = sess["last_assistant_id"]

        if sess["chat_id"] is None:
            chat_id, _ = create_chat(token, prompt, model=chat_model, cookie=cookie,
                                     enable_thinking=req.enable_thinking,
                                     reasoning_effort=req.reasoning_effort)
            sess["chat_id"] = chat_id
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
                    pid = inner.get("parent_id")
                    if mid:
                        new_assistant_id = mid
                    if pid:
                        new_assistant_id = pid
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
                if text:
                    sess["history"].append({"role": "assistant", "content": text})
                    sess["last_assistant_id"] = new_assistant_id or sess["last_assistant_id"]
                    if len(sess["history"]) > MAX_HISTORY * 2:
                        sess["history"] = sess["history"][-MAX_HISTORY * 2:]
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