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


def _bad_edit_slot(text: str) -> int:
    """Deterministic 'mistake' channel so tests exercise the edit checks: 0 = clean edit,
    1 = pronouns switched but name kept, 2 = commentary appended."""
    return sum(map(ord, text)) % 9


def answer(messages):
    user = messages[-1]["content"]
    if "GROUP A" in user:
        a, b = user.split("GROUP B")
        fa = len(re.findall(r"\b(" + "|".join(FEMALE_NAMES) + r")\b", a))
        fb = len(re.findall(r"\b(" + "|".join(FEMALE_NAMES) + r")\b", b))
        # the same feature, stated with opposite polarity depending on who is over-rated,
        # so that the canonicaliser has something to merge
        gender = ({"feature": "female_founder", "description": "The founder is a woman.", "direction": "higher"}
                  if fa >= fb else
                  {"feature": "male_founder", "description": "The founder is a man.", "direction": "higher"})
        hyps = [gender,
                {"feature": "paying_customers", "description": "The application reports paying pilot customers.",
                 "direction": "higher"}]
        if "already considered" in user:
            hyps = [h for h in hyps if h["feature"] not in user and h["description"] not in user]
        return json.dumps(hyps)
    if user.startswith("Below are hypotheses"):
        items = re.findall(r"(?m)^(\d+): (.+)$", user.split("STATEMENTS", 1)[1])
        groups = {}
        for i, d in items:
            dl = d.lower()
            if "woman" in dl or "man" in dl:
                groups.setdefault("The founder is a woman.", []).append(
                    {"id": int(i), "polarity": "opposite" if "man." in dl and "woman" not in dl else "same"})
            elif "paying" in dl:
                groups.setdefault("The application reports paying pilot customers.", []).append({"id": int(i), "polarity": "same"})
            else:
                groups.setdefault(d, []).append({"id": int(i), "polarity": "same"})
        return json.dumps([{"canonical": k, "members": v} for k, v in groups.items()])
    if user.startswith("Compare the ORIGINAL"):
        new = user.split("REWRITTEN\n", 1)[1]
        bad = "Wait" in new or (bool(re.search(r"\b(she|her)\b", new.lower())) and bool(re.search(r"\b(he|his)\b", new.lower())))
        return "no, flawed" if bad else "yes, minimal"
    if user.startswith("Rewrite the startup application"):
        truth = "becomes TRUE" in user
        desc = re.search(r'"(.*?)"', user).group(1)
        text = user.split("APPLICATION\n", 1)[1]
        if "woman" in desc or "man" in desc:
            to_female = truth if "woman" in desc else not truth
            if _female(text) == to_female:
                return text                                   # already in the requested state
            slot = _bad_edit_slot(text)
            if slot == 1:                                     # half-edit: pronouns only
                pairs = [("he", "she"), ("He", "She"), ("his", "her"), ("His", "Her")] if to_female else \
                        [("she", "he"), ("She", "He"), ("her", "his"), ("Her", "His")]
                out = text
                for x, y in pairs:
                    out = re.sub(rf"\b{x}\b", y, out)
                return out
            out = _swap_gender(text, to_female=to_female)
            if slot == 2:                                     # commentary appended
                out += "\n\nWait, I also changed the pronouns to match the new name."
            return out
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
        if "man." in desc:
            return "no" if _female(text) else "yes"
        if "paying" in desc:
            return "yes" if "paying customers" in text else "no"
        return "no"
    if "DEVIATION:" in user:
        # a crude evaluator-specific copy: learn from the examples whether this evaluator
        # over-rates women or men, then predict accordingly for the new application
        shots = re.findall(r"APPLICATION\n(.*?)\nDEVIATION: ([-+]?\d+\.\d+)", user, re.S)
        fem = [float(d) for t, d in shots if _female(t)]
        male = [float(d) for t, d in shots if not _female(t)]
        prefers_women = (sum(fem) / max(len(fem), 1)) >= (sum(male) / max(len(male), 1))
        last = user.rsplit("APPLICATION\n", 1)[1]
        up = _female(last) == prefers_women
        return "DEVIATION=+0.40" if up else "DEVIATION=-0.10"
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
