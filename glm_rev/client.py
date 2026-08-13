"""High-level client operations: session creation, token refresh, and chat streaming."""
import json
import time
import uuid
import requests

from .config import BASE, ENDPOINT
from .api import headers, sign, user_id_from_token

# Features payload verified live against chat.z.ai (glm-5.2) on 2026-08-13:
# enable_thinking=True  -> {"enable_thinking": true, "reasoning_effort": "high"|"max"}
# enable_thinking=False -> {"enable_thinking": false}, reasoning_effort omitted entirely
def build_features(enable_thinking: bool, reasoning_effort: str, web_search: bool = False) -> dict:
    feats = {
        "auto_web_search": False,
        "flags": [],
        "image_generation": False,
        "preview_mode": True,
        "vlm_tools_enable": False,
        "vlm_web_search_enable": False,
        "vlm_website_mode": False,
        "web_search": web_search,
    }
    if enable_thinking:
        feats["enable_thinking"] = True
        if reasoning_effort in ("high", "max"):
            feats["reasoning_effort"] = reasoning_effort
    else:
        feats["enable_thinking"] = False
    return feats


def refresh_token(token: str) -> str:
    r = requests.get(f"{BASE}/api/v1/auths/", headers=headers(token), timeout=20)
    if r.status_code != 200:
        return token
    j = r.json()
    if isinstance(j, dict) and j.get("token"):
        return j["token"]
    return token


def create_chat(token: str, prompt: str, model: str = "glm-5.2", cookie: str = None,
                chat_id: str = None, msg_id: str = None,
                enable_thinking: bool = True, reasoning_effort: str = "max") -> tuple[str, str]:
    chat_id = chat_id or str(uuid.uuid4())
    msg_id = msg_id or str(uuid.uuid4())
    ts2 = int(time.time())
    chat_body = {
        "chat": {
            "id": chat_id,
            "title": "New Chat",
            "models": [model],
            "params": {},
            "history": {
                "messages": {
                    msg_id: {
                        "id": msg_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": prompt,
                        "timestamp": ts2,
                        "models": [model],
                    }
                },
                "currentId": msg_id,
            },
            "tags": [],
            "flags": [],
            "features": [{"server": "tool_selector_h", "status": "hidden", "type": "tool_selector"}],
            "mcp_servers": [],
            "enable_thinking": enable_thinking,
            "reasoning_effort": reasoning_effort if enable_thinking else None,
            "auto_web_search": False,
            "message_version": 1,
            "extra": {},
            "timestamp": int(time.time() * 1000),
            "type": "default",
        }
    }
    r = requests.post(f"{BASE}/api/v1/chats/new", json=chat_body,
                      headers=headers(token, cookie=cookie), timeout=20)
    if r.status_code == 200:
        try:
            chat_id = r.json().get("id", chat_id)
        except Exception:
            pass
    return chat_id, msg_id


def chat(prompt: str, token: str, model: str = "glm-5.2", chat_id: str = None,
         session_id: str = None, stream: bool = True, captcha: str = None,
         cookie: str = None, create_session: bool = True,
         enable_thinking: bool = True, reasoning_effort: str = "max") -> None:

    token = refresh_token(token)
    msg_id = None
    if create_session:
        chat_id, msg_id = create_chat(token, prompt, model, cookie, chat_id,
                                      enable_thinking=enable_thinking,
                                      reasoning_effort=reasoning_effort)
    sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                               current_url=f"{BASE}/chat/{chat_id}")
    url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"
    session_id = session_id or str(uuid.uuid4())
    msg_id = msg_id or str(uuid.uuid4())
    body = {
        "background_tasks": {"title_generation": True, "tags_generation": True},
        "chat_id": chat_id,
        "current_user_message_id": msg_id,
        "current_user_message_parent_id": None,
        "extra": {},
        "features": build_features(enable_thinking, reasoning_effort),
        "id": str(uuid.uuid4()),
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "params": {"max_tokens": 8192, "temperature": 1, "top_p": 0.95},
        "signature_prompt": prompt,
        "stream": stream,
        "variables": {},
    }
    if captcha:
        body["captcha_verify_param"] = captcha
    r = requests.post(url, headers=headers(token, sig, cookie), json=body,
                      stream=True, timeout=120)
    print("HTTP", r.status_code)
    print("CT", r.headers.get("content-type"))
    if r.status_code != 200:
        print(r.text[:2000])
        return
    if not stream:
        print(r.text[:4000])
        return
    full = []
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                break
            full.append(data)
            try:
                j = json.loads(data)
                c = j.get("data", {})
                if c.get("phase") == "thinking":
                    continue
                delta = c.get("delta_content") or ""
                if delta:
                    print(delta, end="", flush=True)
            except json.JSONDecodeError:
                pass
    print()
    for d in full:
        try:
            j = json.loads(d)
            u = j.get("data", {}).get("usage")
            if u:
                print("USAGE:", json.dumps(u))
        except Exception:
            pass


def list_models(token: str) -> None:
    r = requests.get(f"{BASE}/api/models", headers=headers(token, "x"), timeout=30)
    print("HTTP", r.status_code)
    if r.status_code == 200:
        for m in r.json().get("data", []):
            print(m.get("id"))
