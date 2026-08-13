"""GLM web server: OpenAI-compatible API + multi-turn chat endpoint + web UI."""
import asyncio
import json
import os
import re
import time
import uuid
import uvicorn
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union

from glm_rev.solver import steal_captcha, CaptchaSolver
from glm_rev.client import create_chat, refresh_token, build_features
from glm_rev.config import BASE, ENDPOINT
from glm_rev.api import sign, headers, user_id_from_token

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="GLM Proxy + UI")

# Aliyun captcha tokens are SINGLE-USE — never cache or reuse them.
# A warm Playwright solver (daemon thread) issues a fresh token per request.
_warm_solver = None
_captcha_lock = None

# Multi-turn sessions for /api/chat
SESSIONS = {}
MAX_HISTORY = 60


def get_token() -> str:
    try:
        return open(os.path.join(HERE, "zai", "token.txt")).read().strip()
    except Exception:
        raise HTTPException(status_code=500, detail="ZAI token not found in zai/token.txt")


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
            return out
    except Exception:
        pass
    fallback = [
        "glm-5.2", "GLM-5.1", "GLM-5-Turbo", "GLM-5v-Turbo", "glm-4.7",
        "glm-4.6v", "0727-106B-API", "0727-360B-API",
        "GLM-4.1V-Thinking-FlashX", "deep-research", "zero",
        "glm-4-flash", "0808-360B-DR", "glm-4-air-250414",
    ]
    return [{"id": m, "object": "model", "owned_by": "z.ai", "created": 0, "name": m} for m in fallback]


class Message(BaseModel):
    role: str
    content: Union[str, List[dict]]


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


async def acquire_captcha(token: str):
    """Return a fresh (captcha, cookie) pair. Single-use tokens: never reused.

    Tries the warm Playwright solver first (no browser relaunch per request);
    falls back to up to 3 one-shot headless solves. Requests are serialized on
    a lock because the warm solver is a single shared browser."""
    global _captcha_lock
    if _captcha_lock is None:
        _captcha_lock = asyncio.Lock()
    async with _captcha_lock:
        solver = await asyncio.to_thread(_get_warm_solver)
        try:
            if not solver._thread or not solver._thread.is_alive():
                captcha, cookie = await asyncio.to_thread(solver.start, token)
            else:
                captcha, cookie = await asyncio.to_thread(solver.solve, token)
            if captcha:
                return captcha, cookie
        except Exception as e:
            print(f"[!] Warm captcha solver unavailable: {e}")
        for attempt in range(3):
            try:
                captcha, cookie = await steal_captcha(token)
                if captcha:
                    return captcha, cookie
            except Exception as e:
                print(f"[!] One-shot captcha solver attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1)
    raise HTTPException(status_code=502,
                        detail="Failed to acquire Aliyun captcha verification token after multiple attempts")


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": model_catalog()}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    try:
        token = get_token()
        token = refresh_token(token)

        tool_contract_prefix = ""
        if req.tools:
            tool_contract_prefix = (
                "CRITICAL SYSTEM INSTRUCTION:\n"
                "You have access to tools. When you want or need to invoke a tool, you MUST reply with "
                "EXACTLY ONE line in this precise format and nothing else on that line:\n"
                "TOOL:tool_name({\"param\": \"value\"})\n\n"
                "Available tools:\n"
            )
            for t in req.tools:
                fn = t.get("function", {})
                name = fn.get("name")
                desc = fn.get("description", "")
                params = json.dumps(fn.get("parameters", {}))
                tool_contract_prefix += f"- TOOL:{name}({params}) : {desc}\n"
            tool_contract_prefix += "\nAlways use TOOL: format when tool assistance is required.\n\n"

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
        if tool_contract_prefix:
            prompt = tool_contract_prefix + prompt
            formatted_messages[-1]["content"] = prompt

        captcha, cookie = await acquire_captcha(token)

        chat_model = req.model
        if not chat_model:
            chat_model = "glm-5.2"

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
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    delta = j.get("data", {}).get("delta_content") or ""
                    full_text += delta
                except Exception:
                    pass
            return full_text

        if not req.stream:
            full_text = collect()
            tool_match = re.search(r"TOOL:\s*([A-Za-z0-9_-]+)\s*\((.*?)\)", full_text, re.DOTALL)
            if tool_match and req.tools:
                t_name = tool_match.group(1)
                t_raw_args = tool_match.group(2).strip()
                try:
                    t_args = json.loads(t_raw_args)
                except Exception:
                    t_args = {}
                clean_content = full_text.replace(tool_match.group(0), "").strip()
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": clean_content if clean_content else None,
                            "tool_calls": [{
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {"name": t_name, "arguments": json.dumps(t_args)},
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        async def event_generator():
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        j = json.loads(data)
                        delta = j.get("data", {}).get("delta_content") or ""
                        chunk = {
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
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception:
                        pass
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
        chat_model = req.model or "glm-5.2"
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