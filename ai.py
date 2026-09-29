"""Gemini client: ask for answers as an INDEXED JSON list.

Every answer carries the question index it belongs to, so a missing or
extra answer can never shift onto the wrong question.

Designed to be extremely frugal with API quota (the free tier allows only
a few requests per minute per model):
- The (model, search, output-mode) combination that last produced answers
  is remembered and tried FIRST, so a warm fill costs exactly 1 request.
- Models that reject JSON mode or the search tool are remembered per-model
  and skipped forever after — no repeated 400s.
- Search grounding (when on) goes straight to plain-text output, since
  schema/JSON-mode + tools combinations are usually unsupported.
- Transient errors (429/5xx) are retried with backoff, honouring Google's
  own "Please retry in Xs" hint; an overloaded model falls back to a
  lighter text model; retired model names are auto-replaced.

Alternative provider: set AI_PROVIDER=openai plus OPENAI_API_KEY /
OPENAI_BASE_URL / OPENAI_MODEL to use any OpenAI-compatible endpoint
(Groq, OpenRouter, Cerebras, Together, a local Ollama...). Grounding is
Gemini-only, so search is skipped there.
"""

import json
import os
import re
import time

from pydantic import BaseModel

_PROVIDER = (os.getenv("AI_PROVIDER") or "gemini").strip().lower()
IS_COMPAT = _PROVIDER in ("openai", "openai-compat", "compat", "custom")

DEFAULT_MODEL = "gemini-3.6-flash"
if IS_COMPAT:
    MODEL = (os.getenv("OPENAI_MODEL") or "llama-3.3-70b-versatile").strip()
else:
    MODEL = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
FALLBACK_MODEL = (os.getenv("GEMINI_FALLBACK_MODEL") or "").strip()
# Google-Search grounding: lets the model look things up on the live web
# (current affairs / GK / anything past its training cutoff).
WEB_SEARCH = os.getenv("WEB_SEARCH", "1").strip().lower() not in ("0", "false", "no")

RETRYABLE_CODES = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = (2, 4, 8)            # transient server errors
QUOTA_BACKOFF_SECONDS = (10, 20, 30)    # 429 without an explicit retry hint
MAX_ATTEMPTS = 3
RETRY_DELAY_CAP = 60.0

SEARCH_HINTS = ("google_search", "search grounding", "grounding with google",
                "tools are not", "tool use is not")

# model-name parts that mark non-text models (never use these for answers)
NON_TEXT_PARTS = ("tts", "audio", "image", "imagen", "veo", "embed",
                  "live", "aqa", "computer-use")

PROMPT_TEMPLATE = """You are filling out a Google Form on behalf of a user.

USER PROFILE:
{profile}

FORM: {form_title}

QUESTIONS (answer using the question index):
{questions}

Rules:
- Answer EVERY question on the form — do not skip any.
- Maths, logic, reasoning, and general-knowledge questions are NOT
  personal questions: always answer them, working them out yourself
  step by step if needed.
- Personal questions (name, class, roll number, contact...) come from
  the user profile; if the profile truly doesn't cover one, omit only
  that question.
- multiple_choice / dropdown / grid: answer with the EXACT text of one option.
- checkboxes: answer with option texts separated by commas.
- linear_scale: answer with a number between the shown bounds.
- date: answer as YYYY-MM-DD. time: answer as HH:MM (24h).
- For current events or facts you are unsure of, use the web search
  tool (if available) and answer from what you find.
- Respond with ONLY the JSON array — no explanations, no markdown.
"""


class AnswerItem(BaseModel):
    index: int
    answer: str


def render_questions(questions) -> str:
    lines = []
    for q in questions:
        line = f"[{q.index}] ({q.qtype}) {q.label}"
        if getattr(q, "required", False):
            line += " (required)"
        if q.qtype in ("multiple_choice", "dropdown", "checkboxes", "grid"):
            line += f" — options: {q.options}"
        if q.qtype == "linear_scale":
            line += f" — scale {q.scale_min} to {q.scale_max}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(profile: str, form_title: str, questions) -> str:
    return PROMPT_TEMPLATE.format(
        profile=profile.strip() or "(no profile provided)",
        form_title=form_title or "(untitled form)",
        questions=render_questions(questions),
    )


def _loads(text: str):
    """Parse JSON that may be wrapped in a markdown fence or prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]|\{.*\}", t, re.DOTALL)   # embedded JSON
        if m:
            return json.loads(m.group(0))
        raise


def _to_map(items) -> dict:
    """Normalize the many shapes models/gateways return into
    {question_index: answer_string}.

    Accepted shapes:
    - [{"index": 0, "answer": "..."}, ...]            (what we ask for)
    - {"answers": [...]} / {"result": ...} wrappers
    - {"0": "...", "1": "..."}                         (index-keyed object)
    - ["answer1", "answer2", ...]                       (bare positional array)
    - [[0, "answer"], ...]                              (pairs)
    - items with alternate keys (question/value/...) or missing index
    - pydantic-ish objects with .index/.answer
    """
    # unwrap common objects
    if isinstance(items, dict):
        idx_keys = ("index", "question_index", "q_index", "question",
                    "idx", "i", "number", "n")
        ans_keys = ("answer", "value", "a", "text", "response")
        if (any(k in items for k in idx_keys)
                and any(k in items for k in ans_keys)):
            items = [items]              # a single answer object
        else:
            for key in ("answers", "result", "results", "data", "items"):
                v = items.get(key)
                if isinstance(v, (list, dict)):
                    items = v
                    break
    if isinstance(items, dict):
        out = {}
        for k, v in items.items():          # {"0": "answer", ...}
            try:
                out[int(str(k).strip())] = str(v)
            except (TypeError, ValueError):
                continue
        return out
    if not isinstance(items, list):
        raise ValueError("response is neither a JSON array nor an object")

    # bare positional array of scalars: answer i belongs to question i
    if items and all(isinstance(x, (str, int, float)) for x in items):
        return {i: str(x) for i, x in enumerate(items)}

    out = {}
    for pos, item in enumerate(items):
        if isinstance(item, dict):
            idx = None
            for key in ("index", "question_index", "q_index", "question",
                        "idx", "i", "number", "n"):
                if key in item:
                    idx = item[key]
                    break
            ans = None
            for key in ("answer", "value", "a", "text", "response"):
                if key in item:
                    ans = item[key]
                    break
            if idx is None and ans is not None:
                idx = pos                   # fall back to position
            if idx is not None and ans is not None:
                try:
                    out[int(idx)] = str(ans)
                except (TypeError, ValueError):
                    continue
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                out[int(item[0])] = str(item[1])
            except (TypeError, ValueError):
                continue
        elif hasattr(item, "index") and hasattr(item, "answer"):
            try:
                out[int(item.index)] = str(item.answer)
            except (TypeError, ValueError):
                continue
    return out


def _status_code(e) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


# ---------------------------------------------------------------- provider
# ---------------------------------------------------------------- model choice

_resolved_model = None
_available = None        # cached list of model names visible to the account
_working_model = None    # last model that actually produced answers
_last_good = None        # {"model", "search", "kind"} that last produced answers
_no_json_models = set()  # models that reject response_mime_type JSON mode
_no_search_models = set()  # models that reject the search tool


def _version_key(name: str):
    m = re.search(r"(\d+(?:\.\d+)*)", name)
    return tuple(float(p) for p in m.group(1).split(".")) if m else (0,)


def _list_models(client):
    global _available
    if _available is not None:
        return _available
    names = []
    try:
        for m in client.models.list():
            n = getattr(m, "name", "") or ""
            if n.startswith("models/"):
                n = n[len("models/"):]
            if n:
                names.append(n)
    except Exception:
        names = []
    _available = names
    return names


def _is_text_model(name: str) -> bool:
    n = (name or "").lower()
    return not any(part in n for part in NON_TEXT_PARTS)


def resolve_model(client) -> str:
    """GEMINI_MODEL if usable; otherwise the newest stable flash model the
    account can actually see (survives Google retiring model names)."""
    global _resolved_model
    if _resolved_model:
        return _resolved_model
    names = _list_models(client)
    text_models = [n for n in names if _is_text_model(n)]
    if not names or MODEL in names:
        _resolved_model = MODEL
        return MODEL
    flashes = [n for n in text_models if "flash" in n]
    stable = [n for n in flashes
              if "exp" not in n and "preview" not in n and "lite" not in n]
    pool = stable or flashes or text_models or names
    _resolved_model = max(pool, key=_version_key)
    return _resolved_model


def _auto_lite(client) -> str | None:
    """A lighter model to try when the primary is overloaded (503)."""
    lites = [n for n in _list_models(client) if _is_text_model(n)
             and "flash" in n and "lite" in n
             and "exp" not in n and "preview" not in n]
    return max(lites, key=_version_key) if lites else None


def _model_chain(client) -> list:
    """Order to try models in: last one that worked, then the configured /
    resolved primary, then the fallback (explicit env or auto flash-lite)."""
    models = []
    for m in (_working_model, resolve_model(client),
              FALLBACK_MODEL or _auto_lite(client)):
        if m and m not in models:
            models.append(m)
    return models


def _retry_delay(e) -> float | None:
    """Google often says 'Please retry in 27.03s' on 429s — use it."""
    m = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE)
    if m:
        return min(float(m.group(1)), RETRY_DELAY_CAP)
    return None


# ---------------------------------------------------------------- the call

class _SearchUnsupported(Exception):
    def __init__(self, cause):
        super().__init__("search not supported here")
        self.cause = cause


def _search_tool(types):
    try:
        return types.Tool(google_search=types.GoogleSearch())
    except Exception:
        return types.Tool(google_search={})


def _build_configs(types, search: bool, model: str):
    """Returns [(kind, config), ...] in try-order.

    With search grounding on, plain-text output goes first (schema/JSON-mode
    combined with tools is usually rejected). The last combination that
    worked for this model is always tried first."""
    tool = None
    if search and model not in _no_search_models:
        tool = _search_tool(types)
    if tool is not None:
        kinds = ["plain"]
    elif model in _no_json_models:
        kinds = ["plain"]
    else:
        kinds = ["schema", "json", "plain"]

    if (_last_good and _last_good["model"] == model
            and _last_good["search"] == bool(tool)
            and _last_good["kind"] in kinds):
        kinds.remove(_last_good["kind"])
        kinds.insert(0, _last_good["kind"])

    cfgs = []
    for kind in kinds:
        kw = {"temperature": 0.1}
        if kind == "schema":
            kw["response_schema"] = list[AnswerItem]
            kw["response_mime_type"] = "application/json"
        elif kind == "json":
            kw["response_mime_type"] = "application/json"
        if tool is not None:
            kw["tools"] = [tool]
        cfgs.append((kind, types.GenerateContentConfig(**kw)))
    return cfgs


def _attempt(client, models, prompt, search: bool, types):
    """Run the model/config/retry ladder. Raises _SearchUnsupported when the
    API rejects the search tool combination, so the caller can retry without
    search; otherwise returns the answer map or raises the last error."""
    global _working_model, _last_good
    last_err = None
    for model in models:
        overloaded = False
        for kind, cfg in _build_configs(types, search, model):
            attempt = 0
            while True:
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt, config=cfg
                    )
                    _working_model = model
                    _last_good = {"model": model, "search": search, "kind": kind}
                    try:
                        parsed = getattr(resp, "parsed", None)
                        items = parsed if parsed else _loads(resp.text)
                        result = _to_map(items)
                    except Exception as e:
                        raw = repr(getattr(resp, "text", ""))[:400]
                        print(f"FormFill: unparseable AI response ({e}); raw: {raw}")
                        raise ValueError(
                            f"AI response was not in the expected format ({e})"
                        ) from e
                    if not result and items:
                        raw = repr(getattr(resp, "text", ""))[:400]
                        print(f"FormFill: AI response had no usable answers; "
                              f"raw: {raw}")
                        raise ValueError("AI response contained no usable answers")
                    return result
                except Exception as e:
                    last_err = e
                    code = _status_code(e)
                    if code not in RETRYABLE_CODES:
                        msg = str(e).lower()
                        if "json mode" in msg:
                            _no_json_models.add(model)
                        if any(h in msg for h in SEARCH_HINTS):
                            _no_search_models.add(model)
                            raise _SearchUnsupported(e)
                        break            # not transient -> try next config
                    attempt += 1
                    if attempt >= MAX_ATTEMPTS:
                        # model is overloaded / out of quota — another
                        # config won't help, but another model might.
                        overloaded = True
                        break
                    delay = _retry_delay(e)
                    if delay is None:
                        base = QUOTA_BACKOFF_SECONDS if code == 429 else BACKOFF_SECONDS
                        delay = base[min(attempt - 1, len(base) - 1)]
                    time.sleep(delay)
            if overloaded:
                break                    # try the next model instead
        if not overloaded:
            break                        # non-transient: don't try other models
    raise last_err or RuntimeError("Gemini call failed")


def generate_answers(client, profile: str, form_title: str, questions,
                     use_search: bool | None = None) -> dict:
    """Call the AI; returns {question.index: answer_string}.

    use_search: enable Google-Search grounding (default: WEB_SEARCH env,
    on). Not available on OpenAI-compatible providers — skipped there.
    If the model/API rejects the search tool, everything is retried once
    without it, so a fill never fails just because of search.
    """
    global _working_model, _last_good
    from google.genai import types

    prompt = build_prompt(profile, form_title, questions)
    models = _model_chain(client)
    if use_search is None:
        use_search = WEB_SEARCH
    if IS_COMPAT or isinstance(client, OpenAICompatClient):
        use_search = False               # grounding is Gemini-only

    last = None
    for search in ((True, False) if use_search else (False,)):
        try:
            return _attempt(client, models, prompt, search, types)
        except _SearchUnsupported as e:
            last = e.cause               # retry the whole chain without search
            continue
    raise last or RuntimeError("AI call failed")


# ---------------------------------------------------------------- key rotation

class _RotatingModels:
    def __init__(self, rot):
        self._r = rot

    def list(self):
        return self._r._entries[self._r._i][0].models.list()

    def generate_content(self, model=None, **kw):
        """One logical request: try the current entry; on a retryable error
        (429/5xx) immediately try the NEXT entry — no sleeping first. On
        success, advance the rotation so load spreads evenly. Each entry
        that carries its own model name overrides the requested one, so a
        pool can mix providers (Gemini keys + third-party gateways)."""
        last = None
        for _ in range(self._r.size):
            client, own_model = self._r._entries[self._r._i]
            try:
                resp = client.models.generate_content(model=own_model or model, **kw)
                self._r._advance()
                return resp
            except Exception as e:
                last = e
                if _status_code(e) in RETRYABLE_CODES:
                    self._r._advance()
                    continue
                raise                   # not quota/overload: same on any key
        raise last


class RotatingClient:
    """Round-robins several API clients. Entries are either plain clients
    (use whatever model name the caller passes) or (client, model) pairs —
    the entry's own model always wins. All the retry / fallback logic in
    generate_answers sits on top and works unchanged."""

    def __init__(self, clients):
        self._entries = [
            c if isinstance(c, tuple) else (c, None) for c in clients
        ]
        if not self._entries:
            raise ValueError("RotatingClient needs at least one client")
        self._i = 0
        self.models = _RotatingModels(self)

    @property
    def size(self):
        return len(self._entries)

    def _advance(self):
        self._i = (self._i + 1) % len(self._entries)


# ---------------------------------------------------------------- OpenAI-compatible provider

class _CompatModels:
    def __init__(self, client):
        self._c = client

    def list(self):
        return []

    def generate_content(self, model, contents, config):
        return self._c._generate(model, contents, config)


class OpenAICompatClient:
    """Mimics the google-genai client surface (client.models.generate_content)
    against any OpenAI-compatible /chat/completions endpoint: Groq,
    OpenRouter, Cerebras, Together, a local Ollama, ..."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.timeout = timeout
        self.models = _CompatModels(self)

    def _generate(self, model, contents, config):
        import requests

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
        }
        temp = getattr(config, "temperature", None)
        if temp is not None:
            payload["temperature"] = temp
        if getattr(config, "response_schema", None) or \
                getattr(config, "response_mime_type", None):
            payload["response_format"] = {"type": "json_object"}

        r = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message", "") or r.text
            except Exception:
                msg = r.text
            err = RuntimeError(f"{r.status_code} {str(msg)[:300]}")
            err.status_code = r.status_code
            raise err
        data = r.json()
        text = data["choices"][0]["message"]["content"]

        class _Resp:
            parsed = None
        resp = _Resp()
        resp.text = text
        return resp
