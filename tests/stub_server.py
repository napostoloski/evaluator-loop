"""A tiny OpenAI-compatible server that answers the loop's prompts with simple rules.

It exists only to test the plumbing of evloop.llm (requests, parsing, caching, logging and
the loop's control flow) without a real model. It is not a model of anything.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from evloop.testbed import FEMALE_NAMES, MALE_NAMES

F2M = dict(zip(FEMALE_NAMES, MALE_NAMES))
M2F = dict(zip(MALE_NAMES, FEMALE_NAMES))


def _swap_gender(text: str, to_female: bool) -> str:
    table = M2F if to_female else F2M
    for a, b in table.items():
        text = re.sub(rf"\b{a}\b", b, text)
    pairs = [("he", "she"), ("He", "She"), ("his", "her"), ("His", "Her")] if to_female else \
            [("she", "he"), ("She", "He"), ("her", "his"), ("Her", "His")]
    for a, b in pairs:
        text = re.sub(rf"\b{a}\b", b, text)
    return text


def _female(text: str) -> bool:
    return bool(re.search(r"\b(" + "|".join(FEMALE_NAMES) + r")\b", text))


def answer(messages):
    user = messages[-1]["content"]
    if "GROUP A" in user:
        a, b = user.split("GROUP B")
        fa = len(re.findall(r"\b(" + "|".join(FEMALE_NAMES) + r")\b", a))
        fb = len(re.findall(r"\b(" + "|".join(FEMALE_NAMES) + r")\b", b))
        hyps = [{"feature": "female_founder", "description": "The founder is a woman.",
                 "direction": "higher" if fa >= fb else "lower"},
                {"feature": "paying_customers", "description": "The application reports paying pilot customers.",
                 "direction": "higher"}]
        if "already considered" in user:
            hyps = [h for h in hyps if h["feature"] not in user and h["description"] not in user]
        return json.dumps(hyps)
    if user.startswith("Rewrite the startup application"):
        truth = "becomes TRUE" in user
        desc = re.search(r'"(.*?)"', user).group(1)
        text = user.split("APPLICATION\n", 1)[1]
        if "woman" in desc:
            return _swap_gender(text, to_female=truth)
        if "paying" in desc:
            if truth:
                return text if "paying customers" in text else text.replace(
                    "\n\nTarget customer:", " We have already run a pilot with 10 paying customers.\n\nTarget customer:", 1)
            return re.sub(r"[^.]*paying customers[^.]*\.?", "", text)
        return text
    if user.startswith("Does the following statement"):
        desc = re.search(r'"(.*?)"', user).group(1)
        text = user.split("APPLICATION\n", 1)[1]
        if "woman" in desc:
            return "yes" if _female(text) else "no"
        if "paying" in desc:
            return "yes" if "paying customers" in text else "no"
        return "no"
    if "DEVIATION:" in user:
        last = user.rsplit("APPLICATION\n", 1)[1]
        return "+0.40" if _female(last) else "-0.10"
    return "[]"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = answer(body["messages"])
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}],
                          "usage": {"prompt_tokens": 0, "completion_tokens": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


def start():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"
