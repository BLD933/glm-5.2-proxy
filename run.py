#!/usr/bin/env python3
"""CLI runner for GLM-5.1/5.2 client."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from glm_rev.solver import steal_captcha
from glm_rev.client import chat, list_models
from glm_rev.ui import run_repl

TOKEN = Path("/home/bld/glm-rev/zai/token.txt").read_text().strip() if os.environ.get("ZAI_TOKEN") is None else os.environ["ZAI_TOKEN"]


def main():
    ap = argparse.ArgumentParser(description="GLM Chat CLI Client")
    ap.add_argument("prompt", nargs="?", default="Hello test", help="Prompt to send")
    ap.add_argument("--model", default="glm-5.2", help="Model name")
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=True,
                    help="Enable Deep Think (default: on)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="Disable Deep Think")
    ap.add_argument("--effort", choices=["high", "max"], default="max", help="Reasoning effort")
    ap.add_argument("--auto", action="store_true", default=True, help="Auto-solve captcha via headless browser")
    ap.add_argument("--list-models", action="store_true", help="List available models")
    ap.add_argument("--repl", action="store_true", help="Launch interactive REPL")
    ap.add_argument("--no-pretty", action="store_true", help="Disable ANSI markdown rendering in REPL")
    ap.add_argument("--debug-sse", action="store_true",
                    help="Print raw SSE events to stderr for every streamed turn")
    a = ap.parse_args()

    if a.list_models:
        list_models(TOKEN)
        return

    if a.repl:
        run_repl(TOKEN, start_model=a.model, no_pretty=a.no_pretty,
                 start_think=a.thinking, debug_sse=a.debug_sse)
        return

    print("Stealing fresh captcha token & cookies via headless browser...")
    captcha, cookie = asyncio.run(steal_captcha(TOKEN))
    if not captcha:
        print("ERROR: Failed to steal captcha token.")
        sys.exit(1)
    print(f"Captcha acquired ({len(captcha)} chars).")

    chat(a.prompt, TOKEN, a.model, captcha=captcha, cookie=cookie,
         enable_thinking=a.thinking, reasoning_effort=a.effort)


if __name__ == "__main__":
    main()
