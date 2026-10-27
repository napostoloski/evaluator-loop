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
import threading
from concurrent.futures import ThreadPoolExecutor
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
                 log_path: str | None = "llm_calls.jsonl", timeout: int = 180, retries: int = 3,
                 extra_headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
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
        self.n_calls = 0
        self.truncations = 0
        self._lock = threading.Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _log(self, rec):
        if self.log_path:
            with self._lock, open(self.log_path, "a") as f:
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
            with self._lock:
                self.n_calls += 1
            if finish == "length" and (not content or (validate is not None and not validate(content))):
                # the reply was cut off before an answer: try once more with a larger budget
                r2 = dict(r); r2[self.max_tokens_field] = max_tokens * 4
                body, latency = self._post(r2)
                choice = body["choices"][0]
                content = strip_think(choice.get("message", {}).get("content"))
                finish = choice.get("finish_reason")
                rec = dict(time=time.time(), latency=latency, key=key, cached=False, request=r2,
                           content=content, finish_reason=finish, usage=body.get("usage"), retry="length")
                self._log(rec)
                if finish == "length" and (not content or (validate is not None and not validate(content))):
                    with self._lock:
                        self.truncations += 1
                    if self.truncations >= 10 and self.truncations > 0.5 * self.n_calls:
                        raise RuntimeError(
                            "Most replies use the whole token limit without a final answer (probably "
                            "reasoning). Turn reasoning off (e.g. --no-thinking on vLLM with Qwen3-family "
                            "models), use a non-reasoning model, or raise the token limits.")
                    return content
            if validate is None or validate(content):
                if path:
                    tmp = path + f".{threading.get_ident()}.tmp"
                    with open(tmp, "w") as f:
                        json.dump(rec, f)
                    os.replace(tmp, path)
                return content
        return content

    def _post(self, req):
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(req).encode()
        delay = 2.0
        # transient statuses: rate limits, server errors and 529 (provider overloaded); these get
        # more attempts and longer waits than other failures, and honour a Retry-After header
        transient = (408, 429, 500, 502, 503, 504, 529)
        attempts = max(self.retries, 8)
        for attempt in range(attempts + 1):
            t0 = time.time()
            wait = delay
            try:
                r = urllib.request.Request(self.base_url + "/chat/completions", data=data,
                                           headers=headers, method="POST")
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode()), round(time.time() - t0, 3)
            except urllib.error.HTTPError as e:
                if e.code not in transient or attempt == attempts:
                    raise
                try:
                    wait = max(wait, float(e.headers.get("retry-after", 0)))
                except (TypeError, ValueError):
                    pass
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected,
                    ConnectionResetError, json.JSONDecodeError):
                if attempt >= self.retries:
                    raise
            time.sleep(min(wait, 120.0) * (0.75 + 0.5 * np.random.random()))
            delay = min(delay * 2, 120.0)


# ---------------------------------------------------------------------------
# Proposer: hypotheses from the applications an evaluator over- and under-rates
# ---------------------------------------------------------------------------
PROPOSE_SYSTEM = (
    "You are a research assistant helping social scientists understand how a single "
    "evaluator judges startup applications. You generate testable hypotheses about "
    "features of the application text. You never guess the evaluator's identity.")

PROPOSE_USER = """One evaluator scored the applications below. Group A received much HIGHER scores from this evaluator than the other evaluators gave them. Group B received much LOWER scores than the other evaluators gave them.

Propose up to {k} hypotheses about features of the application TEXT that could explain why this evaluator scores Group A above Group B. Each feature must be something that could be added to or removed from an application by a small edit (for example a phrase, a fact about the founder, a property of the venture, or the writing style). Each hypothesis must name ONE concrete, checkable property, not a list of alternatives joined by "or" and not a contrast with several other categories. Avoid features that simply restate overall quality.
{excluded}
Return only a JSON list, with no other text. Each element: {{"feature": "<short snake_case name>", "description": "<one sentence stating the feature as a property an application has>", "direction": "higher" or "lower"}}. Use "higher" if the evaluator rewards applications that have the feature and "lower" if the evaluator penalises them.

GROUP A (over-rated by this evaluator)
{group_a}

GROUP B (under-rated by this evaluator)
{group_b}"""

PROPOSE_USER_V2 = """One evaluator scored the applications below. Group A received much HIGHER scores from this evaluator than the other evaluators gave them. Group B received much LOWER scores than the other evaluators gave them.

Propose {k} DISTINCT hypotheses about features of the application TEXT that could explain why this evaluator scores Group A above Group B. Compare the groups systematically along each of these dimensions before proposing: (1) the founder: gender, name, background, education, experience, team; (2) the venture: sector, product, location, how and where it operates; (3) the evidence offered: customers, pilots, numbers, research; (4) the writing: fluency, grammar, punctuation, polish. Prefer features that are clearly more common in one group than the other. Each feature must be something that could be added to or removed from an application by a small edit, and must name ONE concrete, checkable property, not a list of alternatives. Do not propose the same feature twice in different words. Avoid features that simply restate overall quality.
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
                n_examples: int = 8, excluded: list | None = None, max_chars: int = 2500, prompt: str = "v1"):
    order = np.argsort(resid)
    hi = [texts[i][:max_chars] for i in order[::-1][:n_examples]]
    lo = [texts[i][:max_chars] for i in order[:n_examples]]
    fmt = lambda xs: "\n\n".join(f"(Application {i + 1})\n{t}" for i, t in enumerate(xs))
    ex = ""
    if excluded:
        ex = "Do not propose these features, which were already considered: " + "; ".join(excluded) + "\n"
    msg = [{"role": "system", "content": PROPOSE_SYSTEM},
           {"role": "user", "content": (PROPOSE_USER_V2 if prompt == "v2" else PROPOSE_USER).format(
               k=k, excluded=ex, group_a=fmt(hi), group_b=fmt(lo))}]
    try:
        out = _json_block(client.chat(msg, max_tokens=1200 if prompt == "v1" else 1800, validate=_valid_proposals))
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
    reason: str = ""


_HEADING_RE = re.compile(r"(?m)^([A-Z][A-Za-z /&]{1,40}:)")
# commentary a model may append to or wrap around a rewrite
_COMMENT_RE = re.compile(
    r"(?:^|\n|(?<=[.!?])\s+)(?:Wait\b|Hmm\b|Note:|Notes?:|Explanation:|Here is\b|Here's\b|Rewritten application\b|"
    r"I (?:have|changed|kept|left|will|would|cannot|can't|need|also)\b|The statement (?:is|was) already\b|"
    r"No changes? (?:is |are |was |were )?(?:needed|required|made)\b|\(unchanged\)|\*\*Note)", re.I)


def _headings(text: str) -> list:
    return _HEADING_RE.findall(text)


def _clean_rewrite(new: str, original: str) -> str:
    """Strip reasoning, code fences, preambles and trailing commentary from a rewrite."""
    new = strip_think(new)
    new = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", new.strip())
    heads = _headings(original)
    first_heading = heads[0] if heads else original.split(":", 1)[0] + ":"
    pos = new.find(first_heading)
    if pos > 0:  # drop any preamble before the first section heading
        new = new[pos:]
    m = _COMMENT_RE.search(new)
    if m:        # cut at the first commentary marker; if that removes sections, the sanity check rejects it
        new = new[:m.start()]
    return new.strip()


_FEM_RE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
_MASC_RE = re.compile(r"\b(he|him|his|himself)\b", re.I)


_CAP_STOP = {"She", "He", "Her", "His", "Him", "Hers", "They", "Them", "Their", "The", "We", "Our", "Us", "It", "Its",
             "This", "That", "These", "Those", "In", "At", "On", "For", "With", "And", "But", "Or", "If", "As", "By",
             "To", "From", "Of", "So", "Yet", "Also", "Since", "After", "Before", "Team", "Problem", "Target", "Business"}


def _proper_nouns(text: str) -> set:
    """Capitalised words that are plausibly names: sentence-initial pronouns and function words
    are excluded, so that switching 'He' to 'She' does not count as a name change."""
    return {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text) if w not in _CAP_STOP}


def edit_sanity(original: str, new: str) -> tuple:
    """Free checks on a rewrite: same sections in the same order, no mixed pronouns introduced,
    and a pronoun change must come with a name change. Returns (ok, reason)."""
    if not new:
        return False, "empty"
    ho, hn = _headings(original), _headings(new)
    if ho and hn != ho:
        return False, "sections"
    fo, mo = bool(_FEM_RE.search(original)), bool(_MASC_RE.search(original))
    fn, mn = bool(_FEM_RE.search(new)), bool(_MASC_RE.search(new))
    if fn and mn and not (fo and mo):
        return False, "mixed_pronouns"
    if (fn, mn) != (fo, mo) and _proper_nouns(new) == _proper_nouns(original):
        return False, "pronouns_without_name"
    return True, ""


def word_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def edit_llm(client: LLMClient, text: str, description: str, present: bool,
             min_similarity: float = 0.6) -> EditResult:
    msg = [{"role": "user", "content": EDIT_USER.format(
        truth="TRUE" if present else "FALSE", description=description, text=text)}]
    new = _clean_rewrite(client.chat(msg, max_tokens=1600, validate=lambda t: len(t) > 0.5 * len(text)), text)
    sim = word_similarity(text, new)
    ok, reason = edit_sanity(text, new)
    if ok and sim < min_similarity:
        ok, reason = False, "similarity"
    return EditResult(new, sim, ok, reason)


# ---------------------------------------------------------------------------
# Coder: does an application have a feature?
# ---------------------------------------------------------------------------
CODE_USER = """Does the following statement describe the startup application below? Begin your reply with the single word yes or no.

Statement: "{description}"

APPLICATION
{text}"""


def _yes_no(text):
    m = re.search(r"[a-z]+", strip_think(text).lower())
    return m.group() if m else ""


def code_llm(client: LLMClient, text: str, description: str) -> int:
    reply = client.chat([{"role": "user", "content": CODE_USER.format(description=description, text=text)}],
                        max_tokens=40, validate=lambda t: _yes_no(t) in ("yes", "no"))
    return int(_yes_no(reply) == "yes")


# ---------------------------------------------------------------------------
# Verifier: is a rewrite minimal, complete and free of contradictions?
# ---------------------------------------------------------------------------
VERIFY_USER = """Compare the ORIGINAL and the REWRITTEN startup application. The rewrite was supposed to change the application only so that this statement becomes {truth}:

"{description}"

Answer yes only if all of the following hold: (1) the statement is now {truth} in the rewrite; (2) nothing else of substance changed; (3) the rewrite contains no commentary, notes or explanations; (4) the rewrite has no internal contradiction, such as a name and pronouns of different genders. Begin your reply with the single word yes or no, then one short reason.

ORIGINAL
{original}

REWRITTEN
{new}"""


def verify_edit_llm(client: LLMClient, original: str, new: str, description: str, present: bool) -> bool:
    msg = [{"role": "user", "content": VERIFY_USER.format(
        truth="TRUE" if present else "FALSE", description=description, original=original, new=new)}]
    reply = client.chat(msg, max_tokens=60, validate=lambda t: _yes_no(t) in ("yes", "no"))
    return _yes_no(reply) == "yes"


# ---------------------------------------------------------------------------
# Canonicaliser: merge hypotheses that state the same feature in different words
# ---------------------------------------------------------------------------
CANON_USER = """Below are hypotheses about features of startup applications, each a one-sentence statement, proposed separately for different evaluators. Group the statements that describe the SAME feature, even if worded differently or stated as opposites (for example "the founder is a woman" and "the founder is a man" describe the same feature with opposite polarity). Statements about genuinely different features stay in different groups.

Return only a JSON list of groups. Each group: {{"canonical": "<one sentence stating the feature as a property an application has>", "members": [{{"id": <id>, "polarity": "same" or "opposite"}}, ...]}}. "opposite" means the member statement is the negation of the canonical statement. Every id must appear in exactly one group.

STATEMENTS
{items}"""

_STOP = set(("the a an of or and in to for with is are has have that than rather as by on at from its their it this such e g "
             "application applications states state stated section team founder founders describes described venture "
             "mentions mentioned mention reports includes text currently being holds holding rather lead led").split())


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def canonicalize_offline(descriptions: list, threshold: float = 0.4) -> list:
    """Fallback grouping by word overlap (Jaccard). Cannot detect opposites. Returns, per
    input, (canonical description, flip) with flip always False."""
    groups = []     # list of (words, canonical, [ids])
    out = [None] * len(descriptions)
    for i, d in enumerate(descriptions):
        w = _content_words(d)
        best, best_j = 0.0, None
        for gi, (gw, _, _) in enumerate(groups):
            j = len(w & gw) / max(len(w | gw), 1)
            if j > best:
                best, best_j = j, gi
        if best_j is not None and best >= threshold:
            groups[best_j][2].append(i)
            out[i] = (groups[best_j][1], False)
        else:
            groups.append((w, d, [i]))
            out[i] = (d, False)
    return out


def canonicalize_llm(client: LLMClient, descriptions: list) -> list:
    """Ask the model to group equivalent statements. Returns, per input, (canonical
    description, flip) where flip means the input is the negation of the canonical."""
    if len(descriptions) <= 1:
        return [(d, False) for d in descriptions]
    items = "\n".join(f"{i}: {d}" for i, d in enumerate(descriptions))
    msg = [{"role": "user", "content": CANON_USER.format(items=items)}]

    def valid(t):
        try:
            groups = _json_block(t)
        except ValueError:
            return False
        ids = [m.get("id") for g in groups if isinstance(g, dict) for m in g.get("members", []) if isinstance(m, dict)]
        return len(ids) > 0 and len(ids) == len(set(ids))

    try:
        groups = _json_block(client.chat(msg, max_tokens=2000, validate=valid))
    except ValueError:
        return canonicalize_offline(descriptions)
    out = [None] * len(descriptions)
    for g in groups:
        if not isinstance(g, dict) or not g.get("members"):
            continue
        canon = str(g.get("canonical", "")).strip()
        for m in g["members"]:
            try:
                i = int(m.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= i < len(descriptions) and out[i] is None:
                flip = str(m.get("polarity", "same")).strip().lower().startswith("opp")
                out[i] = (canon or descriptions[i], flip)
    fallback = canonicalize_offline(descriptions)
    return [o if o is not None else fallback[i] for i, o in enumerate(out)]


# ---------------------------------------------------------------------------
# Twin: a few-shot copy of one evaluator's deviations from the panel
# ---------------------------------------------------------------------------
TWIN_SYSTEM = (
    "You are imitating one specific expert evaluator of startup applications for an incubation "
    "program in Pakistan. For each example you see how far this evaluator's score was above (+) "
    "or below (-) the average of the other evaluators. Learn this evaluator's idiosyncrasies and "
    "predict the deviation for a new application, a signed number between -2 and 2. You may reason "
    "briefly, but the last line of your reply must be exactly of the form DEVIATION=<number>, "
    "for example DEVIATION=+0.35 or DEVIATION=-0.20.")


def _last_number(text):
    t = strip_think(text)
    m = re.findall(r"DEVIATION\s*[=:]\s*([-+]?\d+(?:\.\d+)?)", t, flags=re.I)
    if m:
        return float(m[-1])
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
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
        reply = self.client.chat(self._prompt(text), max_tokens=240,
                                 validate=lambda t: re.search(r"DEVIATION\s*[=:]\s*[-+]?\d", strip_think(t), re.I) is not None)
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

    Edits that fail the offline sanity checks (sections changed, pronouns mixed, pronouns
    changed without the name, too dissimilar) are dropped before any query is spent on
    them. Battery edits are additionally checked by the coder (the intended value must now
    read in the text) and, with verify=True, by a verifier call that compares original and
    rewrite. Hypotheses that passed screening are canonicalised across the panel before the
    batteries, so that the same feature stated in different words is pooled.

    twin="llm" uses a few-shot copy of the evaluator; twin="ridge" screens the LLM's edits
    with the ridge text copy from the offline benchmark, which needs no model calls.
    """

    def __init__(self, client: LLMClient, world, n_shots: int = 12, n_twin_pairs: int = 30,
                 min_similarity: float = 0.6, twin: str = "llm", verify: bool = True, workers: int = 4,
                 proposer_prompt: str = "v1", n_examples: int = 8, log=print):
        self.client = client
        self.proposer_prompt = proposer_prompt
        self.n_examples = n_examples
        self.workers = max(1, workers)
        self._lock = threading.Lock()
        self.world = world
        self.texts = [a.text for a in world.apps]
        self.n_shots = n_shots
        self.n_twin_pairs = n_twin_pairs
        self.min_similarity = min_similarity
        self.twin_mode = twin
        self.verify = verify
        self.descriptions = {}
        self.log = log
        self.dropped_edits = 0
        self.drop_reasons = {}
        self.groups = []            # canonical groups formed in each round (for the audit trail)
        self._space = None

    @staticmethod
    def _key(description):
        return re.sub(r"[^a-z0-9]+", " ", description.lower()).strip()

    def _drop(self, reason):
        with self._lock:
            self.dropped_edits += 1
            self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def propose(self, j, resid, excluded, cfg, rng, rnd):
        ex = [self.descriptions.get(f, f) for f in excluded]
        raw = propose_llm(self.client, self.texts, resid, k=cfg.max_hypotheses, excluded=ex,
                          n_examples=self.n_examples, prompt=self.proposer_prompt)
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
        """(text with the feature, text without it) for one application: the original and one
        rewrite in the opposite direction. The coder decides the original's state, so only one
        rewrite is needed, and a rewrite that comes back unchanged is dropped."""
        text = self.texts[i]
        has = code_llm(self.client, text, description)
        e = edit_llm(self.client, text, description, present=not has, min_similarity=self.min_similarity)
        if not e.ok:
            self._drop(e.reason)
            return None
        if e.text.strip() == text.strip():
            self._drop("unchanged")
            return None
        return (text, e.text) if has else (e.text, text)

    def _ridge_space(self, cfg):
        if self._space is None:
            from .loop import TextSpace
            self._space = TextSpace(self.world.apps, seed=cfg.seed, coder_error=cfg.coder_error)
        return self._space

    def screen(self, j, resid, hyps, cfg, rng):
        order = np.argsort(resid)
        shots = np.concatenate([order[: self.n_shots // 2], order[::-1][: self.n_shots - self.n_shots // 2]])
        held = np.array([i for i in rng.permutation(len(self.texts)) if i not in set(shots)])
        if self.twin_mode == "ridge":
            from .loop import RidgeTwin
            ridge = RidgeTwin(self._ridge_space(cfg), resid, cfg, rng)
            twin_effect = lambda pairs: ridge.edit_effect_texts(self._space, [a for a, _ in pairs], [b for _, b in pairs])
        else:
            llm_twin = LLMTwin(self.client, [(self.texts[i], float(resid[i])) for i in shots])
            twin_effect = llm_twin.edit_effect
        for h in hyps:
            idx = [int(i) for i in held[: self.n_twin_pairs]]
            if self.workers > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    results = list(ex.map(lambda i: self._edit_pair(i, h.description), idx))
            else:
                results = [self._edit_pair(i, h.description) for i in idx]
            pairs = [p for p in results if p]
            h.n_pairs = len(pairs)
            h.examples = pairs[:3]
            eff, ci = twin_effect(pairs)
            sign = 1.0 if h.target == 1 else -1.0
            h.twin_effect = sign * eff
            h.twin_ci = tuple(sorted((sign * ci[0], sign * ci[1])))
            self.log(f"evaluator {j} round {h.round}: {h.feature} twin effect {h.twin_effect:+.2f} ({len(pairs)} pairs)")

    def canonicalize(self, cands: dict):
        """Pool hypotheses that state the same feature across evaluators. Rewrites each
        hypothesis's feature key, description and (if its statement is the negation of the
        canonical one) its target. Drops duplicate features within one evaluator, keeping
        the one with the larger predicted effect."""
        items = [(j, h) for j, hs in cands.items() for h in hs]
        if len(items) <= 1:
            return
        mapping = canonicalize_llm(self.client, [h.description for _, h in items])
        for (j, h), (canon, flip) in zip(items, mapping):
            key = self._key(canon) or h.feature
            if key != h.feature:
                self.descriptions.setdefault(key, canon)
                h.feature, h.description = key, self.descriptions[key]
            if flip:
                h.target = 1 - h.target
        groups = {}
        for j, h in items:
            groups.setdefault(h.feature, []).append(j)
        self.groups.append({k: sorted(set(v)) for k, v in groups.items()})
        for k, js in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(js) > 1:
                self.log(f"pooled: '{self.descriptions.get(k, k)[:70]}' requested by evaluators {sorted(set(js))}")
        for j, hs in cands.items():
            best = {}
            for h in hs:
                if h.feature not in best or (h.twin_effect or 0) > (best[h.feature].twin_effect or 0):
                    best[h.feature] = h
            cands[j] = list(best.values())

    def flip_fn(self, feature, description):
        def flip(app):
            has = code_llm(self.client, app.text, description)
            e = edit_llm(self.client, app.text, description, present=not has, min_similarity=self.min_similarity)
            if not e.ok:
                self._drop(e.reason)
                return None
            if e.text.strip() == app.text.strip():
                self._drop("unchanged")
                return None
            if code_llm(self.client, e.text, description) == has:
                self._drop("manipulation_check")
                return None
            if self.verify and not verify_edit_llm(self.client, app.text, e.text, description, present=not has):
                self._drop("verifier")
                return None
            return e.text, (-1 if has else 1)
        return flip

    def fires(self, feature, description, target):
        return np.array([code_llm(self.client, t, description) == target for t in self.texts], dtype=float)
