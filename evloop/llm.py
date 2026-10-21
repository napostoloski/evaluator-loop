"""LLM backend for the loop: proposer, editor, coder and twin.

Works with any server that speaks the OpenAI chat-completions protocol, including a
self-hosted open-weight model (vLLM, Ollama, llama.cpp server, LM Studio) and commercial
APIs. Uses only the Python standard library. Replies are validated before they are cached,
and every call (including cache hits) is appended to a JSONL log, so a run can be audited
and replayed.

    client = LLMClient("http://localhost:8000/v1", "Qwen/Qwen3.8-27B",
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    backend = LLMBackend(client, world)
    results = loop.run_panel(world, backend, LoopConfig())
"""
from __future__ import annotations

import difflib
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .loop import Hypothesis


# ---------------------------------------------------------------------------
# Reply cleaning
# ---------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?(</think>|\Z)", flags=re.S)


def strip_think(text: str | None) -> str:
    """Remove reasoning blocks, including one left open by a truncated reply."""
    return _THINK.sub("", text or "").strip()


def _json_block(text: str):
    """Return the first JSON list in a reply, or the first list inside a JSON object."""
    text = strip_think(text)
    dec = json.JSONDecoder()
    fallback = None
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            inner = [v for v in obj.values() if isinstance(v, list)]
            obj = inner[0] if inner else [obj]
        if obj and all(isinstance(x, dict) for x in obj):
            return obj                      # a list of records: what the prompts ask for
        if fallback is None:
            fallback = obj                  # e.g. "[1]" in prose; keep looking
    if fallback is not None:
        return fallback
    raise ValueError(f"no JSON list in reply: {text[:200]}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class LLMClient:
    """Minimal OpenAI-compatible chat client with validation-gated caching, retries and logs.

    extra_body is merged into every request, e.g. {"chat_template_kwargs":
    {"enable_thinking": False}} for Qwen3-family models on vLLM. For models that reject
    temperature or max_tokens (some reasoning models), pass temperature=None and
    max_tokens_field="max_completion_tokens".
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float | None = 0.0, seed: int | None = 0, extra_body: dict | None = None,
                 max_tokens_field: str = "max_tokens", cache_dir: str | None = "llm_cache",
                 log_path: str | None = "llm_calls.jsonl", timeout: int = 180, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("EVLOOP_API_KEY")
        self.temperature = temperature
        self.seed = seed
        self.extra_body = extra_body or {}
        self.max_tokens_field = max_tokens_field
        self.cache_dir = cache_dir
        self.log_path = log_path
        self.timeout = timeout
        self.retries = retries
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _log(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def chat(self, messages: list, max_tokens: int = 800, validate=None) -> str:
        """Return the reply text (reasoning blocks removed).

        validate(text) -> bool decides whether the reply may be cached. A reply that fails
        validation is retried once and, if it fails again, returned uncached.
        """
        req = dict(model=self.model, messages=messages, **{self.max_tokens_field: max_tokens})
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.seed is not None:
            req["seed"] = self.seed
        req.update(self.extra_body)
        key = hashlib.sha256(json.dumps([self.base_url, req], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key + ".json") if self.cache_dir else None
        if path and os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            self._log(dict(time=time.time(), key=key, cached=True, content=rec["content"]))
            return rec["content"]
        content = ""
        for attempt in range(2):
            r = dict(req)
            if attempt and self.seed is not None:
                r["seed"] = self.seed + 1000      # a different draw; the reply is cached under the original key
            body, latency = self._post(r)
            choice = body["choices"][0]
            raw = choice.get("message", {}).get("content")
            content = strip_think(raw)
            finish = choice.get("finish_reason")
            rec = dict(time=time.time(), latency=latency, key=key, cached=False, request=r,
                       content=content, finish_reason=finish, usage=body.get("usage"))
            self._log(rec)
            if not content and finish == "length":
                raise RuntimeError(
                    "The model used the whole token limit without a final answer (probably reasoning). "
                    "Turn reasoning off (e.g. --no-thinking on vLLM with Qwen3-family models), use a "
                    "non-reasoning model, or raise the token limits.")
            if validate is None or validate(content):
                if path:
                    with open(path, "w") as f:
                        json.dump(rec, f)
                return content
        return content

    def _post(self, req):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(req).encode()
        delay = 2.0
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = urllib.request.Request(self.base_url + "/chat/completions", data=data,
                                           headers=headers, method="POST")
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode()), round(time.time() - t0, 3)
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected,
                    ConnectionResetError, json.JSONDecodeError):
                if attempt == self.retries:
                    raise
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# Proposer: hypotheses from the applications an evaluator over- and under-rates
# ---------------------------------------------------------------------------
PROPOSE_SYSTEM = (
    "You are a research assistant helping social scientists understand how a single "
    "evaluator judges startup applications. You generate testable hypotheses about "
    "features of the application text. You never guess the evaluator's identity.")

PROPOSE_USER = """One evaluator scored the applications below. Group A received much HIGHER scores from this evaluator than the other evaluators gave them. Group B received much LOWER scores than the other evaluators gave them.

Propose up to {k} hypotheses about features of the application TEXT that could explain why this evaluator scores Group A above Group B. Each feature must be something that could be added to or removed from an application by a small edit (for example a phrase, a fact about the founder, a property of the venture, or the writing style). Avoid features that simply restate overall quality.
{excluded}
Return only a JSON list, with no other text. Each element: {{"feature": "<short snake_case name>", "description": "<one sentence stating the feature as a property an application has>", "direction": "higher" or "lower"}}. Use "higher" if the evaluator rewards applications that have the feature and "lower" if the evaluator penalises them.

GROUP A (over-rated by this evaluator)
{group_a}

GROUP B (under-rated by this evaluator)
{group_b}"""

_DIR_UP = {"higher", "high", "up", "positive", "rewards", "reward", "+"}
_DIR_DOWN = {"lower", "low", "down", "negative", "penalises", "penalizes", "penalise", "penalize", "-"}


def _valid_proposals(text):
    try:
        return isinstance(_json_block(text), list)
    except ValueError:
        return False


def propose_llm(client: LLMClient, texts: list, resid: np.ndarray, k: int = 4,
                n_examples: int = 8, excluded: list | None = None, max_chars: int = 2500):
    order = np.argsort(resid)
    hi = [texts[i][:max_chars] for i in order[::-1][:n_examples]]
    lo = [texts[i][:max_chars] for i in order[:n_examples]]
    fmt = lambda xs: "\n\n".join(f"(Application {i + 1})\n{t}" for i, t in enumerate(xs))
    ex = ""
    if excluded:
        ex = "Do not propose these features, which were already considered: " + "; ".join(excluded) + "\n"
    msg = [{"role": "system", "content": PROPOSE_SYSTEM},
           {"role": "user", "content": PROPOSE_USER.format(k=k, excluded=ex, group_a=fmt(hi), group_b=fmt(lo))}]
    try:
        out = _json_block(client.chat(msg, max_tokens=1200, validate=_valid_proposals))
    except ValueError:
        return []
    hyps = []
    for h in out[:k]:
        if not isinstance(h, dict) or "description" not in h:
            continue
        d = str(h.get("direction", "")).strip().lower()
        if d in _DIR_UP:
            target = 1
        elif d in _DIR_DOWN:
            target = 0
        else:
            continue
        name = re.sub(r"[^a-z0-9_]+", "_", str(h.get("feature", "feature")).lower()).strip("_")[:60] or "feature"
        hyps.append(dict(feature=name, description=str(h["description"]).strip(), target=target))
    return hyps


# ---------------------------------------------------------------------------
# Editor: minimal counterfactual rewrites
# ---------------------------------------------------------------------------
EDIT_USER = """Rewrite the startup application below so that the following statement becomes {truth}:

"{description}"

Make the smallest possible change. Keep every other fact, the section headings, the length, the tone and the writing quality exactly as they are. If the statement is already {truth}, return the application unchanged. Return only the rewritten application, with no comments.

APPLICATION
{text}"""


@dataclass
class EditResult:
    text: str
    similarity: float
    ok: bool


def _clean_rewrite(new: str, original: str) -> str:
    new = strip_think(new)
    new = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", new.strip())
    first_heading = original.split(":", 1)[0] + ":"
    pos = new.find(first_heading)
    if pos > 0:  # drop any preamble before the first section heading
        new = new[pos:]
    return new.strip()


def word_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def edit_llm(client: LLMClient, text: str, description: str, present: bool,
             min_similarity: float = 0.6) -> EditResult:
    msg = [{"role": "user", "content": EDIT_USER.format(
        truth="TRUE" if present else "FALSE", description=description, text=text)}]
    new = _clean_rewrite(client.chat(msg, max_tokens=1600, validate=lambda t: len(t) > 0.5 * len(text)), text)
    sim = word_similarity(text, new)
    return EditResult(new, sim, sim >= min_similarity)


# ---------------------------------------------------------------------------
# Coder: does an application have a feature?
# ---------------------------------------------------------------------------
CODE_USER = """Does the following statement describe the startup application below? Answer with one word, yes or no.

Statement: "{description}"

APPLICATION
{text}"""


def _yes_no(text):
    m = re.search(r"[a-z]+", strip_think(text).lower())
    return m.group() if m else ""


def code_llm(client: LLMClient, text: str, description: str) -> int:
    reply = client.chat([{"role": "user", "content": CODE_USER.format(description=description, text=text)}],
                        max_tokens=20, validate=lambda t: _yes_no(t) in ("yes", "no"))
    return int(_yes_no(reply) == "yes")


# ---------------------------------------------------------------------------
# Twin: a few-shot copy of one evaluator's deviations from the panel
# ---------------------------------------------------------------------------
TWIN_SYSTEM = (
    "You are imitating one specific expert evaluator of startup applications for an incubation "
    "program in Pakistan. For each example you see how far this evaluator's score was above (+) "
    "or below (-) the average of the other evaluators. Learn this evaluator's idiosyncrasies and "
    "predict the deviation for a new application. Reply with a single signed number between -2 and 2.")


def _last_number(text):
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", strip_think(text))
    return float(nums[-1]) if nums else None


@dataclass
class LLMTwin:
    client: LLMClient
    examples: list = field(default_factory=list)   # list of (text, deviation from the panel)
    max_chars: int = 2500

    def _prompt(self, text):
        shots = "\n\n".join(f"APPLICATION\n{t[:self.max_chars]}\nDEVIATION: {d:+.2f}" for t, d in self.examples)
        return [{"role": "system", "content": TWIN_SYSTEM},
                {"role": "user", "content": f"{shots}\n\nAPPLICATION\n{text[:self.max_chars]}\nDEVIATION:"}]

    def deviation(self, text: str) -> float:
        reply = self.client.chat(self._prompt(text), max_tokens=20, validate=lambda t: _last_number(t) is not None)
        x = _last_number(reply)
        return float(np.clip(x, -2, 2)) if x is not None else float("nan")

    def edit_effect(self, pairs: list) -> tuple:
        """pairs: (text with feature, text without feature). Mean effect and 95% interval."""
        d = np.array([self.deviation(a) - self.deviation(b) for a, b in pairs])
        d = d[~np.isnan(d)]
        if len(d) < 2:
            return float("nan"), (float("nan"), float("nan"))
        se = d.std(ddof=1) / np.sqrt(len(d))
        return float(d.mean()), (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se))


# ---------------------------------------------------------------------------
# Backend for loop.run_panel
# ---------------------------------------------------------------------------
class LLMBackend:
    """LLM proposer, twin, editor and coder, with the same interface as loop.OfflineBackend.

    Edits that drift too far from the original (word-level similarity below
    min_similarity) or fail a manipulation check (the coder does not read the intended value
    in the edited text) are dropped before any real query is spent on them.
    """

    def __init__(self, client: LLMClient, world, n_shots: int = 12, n_twin_pairs: int = 10,
                 min_similarity: float = 0.6, log=print):
        self.client = client
        self.world = world
        self.texts = [a.text for a in world.apps]
        self.n_shots = n_shots
        self.n_twin_pairs = n_twin_pairs
        self.min_similarity = min_similarity
        self.descriptions = {}
        self.log = log
        self.dropped_edits = 0

    @staticmethod
    def _key(description):
        return re.sub(r"[^a-z0-9]+", " ", description.lower()).strip()

    def propose(self, j, resid, excluded, cfg, rng, rnd):
        # features are keyed by their (normalised) description, so two evaluators' hypotheses are
        # pooled into one battery only if they state the same feature; synonyms are not merged
        ex = [self.descriptions.get(f, f) for f in excluded]
        raw = propose_llm(self.client, self.texts, resid, k=cfg.max_hypotheses, excluded=ex)
        hyps = []
        for h in raw:
            key = self._key(h["description"])
            if not key or key in excluded:
                continue
            self.descriptions.setdefault(key, h["description"])
            hyps.append(Hypothesis(key, h["target"], float("nan"), [h["feature"]], rnd,
                                   description=h["description"]))
        return hyps

    def _edit_pair(self, i, description):
        a = edit_llm(self.client, self.texts[i], description, True, self.min_similarity)
        b = edit_llm(self.client, self.texts[i], description, False, self.min_similarity)
        if not (a.ok and b.ok):
            self.dropped_edits += 1
            return None
        return a.text, b.text

    def screen(self, j, resid, hyps, cfg, rng):
        # contrasting examples: the applications this evaluator most over- and under-rates
        order = np.argsort(resid)
        shots = np.concatenate([order[: self.n_shots // 2], order[::-1][: self.n_shots - self.n_shots // 2]])
        held = np.array([i for i in rng.permutation(len(self.texts)) if i not in set(shots)])
        twin = LLMTwin(self.client, [(self.texts[i], float(resid[i])) for i in shots])
        for h in hyps:
            pairs = [p for p in (self._edit_pair(int(i), h.description) for i in held[: self.n_twin_pairs]) if p]
            eff, ci = twin.edit_effect(pairs)
            sign = 1.0 if h.target == 1 else -1.0
            h.twin_effect = sign * eff
            h.twin_ci = tuple(sorted((sign * ci[0], sign * ci[1])))
            self.log(f"evaluator {j} round {h.round}: {h.feature} twin effect {h.twin_effect:+.2f}")

    def flip_fn(self, feature, description):
        def flip(app):
            has = code_llm(self.client, app.text, description)
            e = edit_llm(self.client, app.text, description, present=not has, min_similarity=self.min_similarity)
            if not e.ok or code_llm(self.client, e.text, description) == has:
                self.dropped_edits += 1
                return None
            return e.text, (-1 if has else 1)
        return flip

    def fires(self, feature, description, target):
        return np.array([code_llm(self.client, t, description) == target for t in self.texts], dtype=float)
