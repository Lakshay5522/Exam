"""FormFill — self-hosted Google Form auto-prefill.

MAGIC URLS (all work is done in the backend; the browser is redirected
straight to the pre-filled Google Form — no loading page):

  http://server/https://docs.google.com/forms/d/e/<id>/viewform   paste format
  http://server/f/<form-id>                                        short format
  http://server/docs.lakshay.google.com/forms/d/e/<id>/viewform    keyword: profile
  http://server/lakshay/https://docs.google.com/forms/...          keyword: profile
  https://lakshay.your-domain.com/forms/...                        hosted wildcard

The keyword (e.g. "lakshay") picks the profile from profiles.json — the
person the form is filled as. Add ?preview=1 to see answers first.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Env:  GEMINI_API_KEY (required), ACCESS_CODE (optional),
      GEMINI_MODEL / GEMINI_FALLBACK_MODEL (optional)
"""

import html as html_mod
import json
import os
import re
import time
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai
from formparser import FormParseError, form_id_from_url, parse_form, viewform_url_for
from prefill import build_prefill

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
WEB_SEARCH = os.getenv("WEB_SEARCH", "1").strip().lower() not in ("0", "false", "no")
PROFILES_PATH = os.path.join(BASE_DIR, "profiles.json")

app = FastAPI(title="FormFill")

_gemini_client = None
_cache: dict = {}          # (form_id, profile_name, overrides, search) -> (ts, result)
CACHE_TTL = 1800           # seconds — keep repeat opens free of API calls


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def check_access(*, header: str | None = None, query: str | None = None):
    if ACCESS_CODE and header != ACCESS_CODE and query != ACCESS_CODE:
        raise HTTPException(401, "Wrong or missing access code (X-Access-Code header or ?code=).")


def _parse_keys(value: str) -> list:
    """Split a comma/newline-separated key list, keeping non-empty parts."""
    return [k.strip() for k in (value or "").replace("\n", ",").split(",")
            if k.strip()]


def _parse_compat_endpoints(raw: str) -> list:
    """Third-party OpenAI-compatible endpoints. Entries separated by ';'
    (or newlines), fields by '|':  name|base_url|api_key|model
    (name optional: base_url|api_key|model)."""
    out = []
    for chunk in (raw or "").replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) == 3:
            parts = ["compat"] + parts
        if len(parts) != 4 or not all(parts):
            print(f"FormFill: ignoring malformed COMPAT_ENDPOINTS entry: "
                  f"{chunk[:60]}... (expected name|base_url|key|model)")
            continue
        out.append(tuple(parts))
    return out


def gemini_client():
    """The AI client pool. Sources, all rotated together with instant
    failover on quota errors:
    - GEMINI_API_KEYS=key1,key2,...  (multiple free Gemini keys)
    - COMPAT_ENDPOINTS=name|base|key|model;...  (third-party OpenAI-
      compatible gateways like codecraftapi.com or tokenbargain.dev)
    - AI_PROVIDER=openai + OPENAI_* (single OpenAI-compatible provider)"""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    entries = []          # list of (client, model_or_None)
    provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()

    if provider in ("openai", "openai-compat", "compat", "custom"):
        keys = _parse_keys(os.getenv("OPENAI_API_KEY", ""))
        if not keys:
            raise HTTPException(500, "AI_PROVIDER=openai but OPENAI_API_KEY is not set.")
        base = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        entries += [(ai.OpenAICompatClient(base, k), ai.MODEL) for k in keys]
    else:
        gem_keys = (_parse_keys(os.getenv("GEMINI_API_KEYS", ""))
                    or _parse_keys(GEMINI_API_KEY))
        if gem_keys:
            from google import genai
            entries += [(genai.Client(api_key=k), None) for k in gem_keys]

    for _name, base, key, model in _parse_compat_endpoints(
            os.getenv("COMPAT_ENDPOINTS", "")):
        entries.append((ai.OpenAICompatClient(base, key), model))

    if not entries:
        raise HTTPException(500, "No API keys configured — set GEMINI_API_KEY(S) "
                                 "or COMPAT_ENDPOINTS in .env")
    if len(entries) == 1:
        _gemini_client = entries[0][0]
    else:
        names = [len(_parse_keys(os.getenv("GEMINI_API_KEYS", "")
                                 or GEMINI_API_KEY)), len(entries)]
        print(f"FormFill: {len(entries)} API endpoints loaded — rotation on "
              f"(quota multiplied by {len(entries)})")
        _gemini_client = ai.RotatingClient(entries)
    return _gemini_client


def load_profiles() -> dict:
    """profiles.json: {"default": "...", "lakshay": "...", ...}. Auto-created.

    On hosting platforms with an ephemeral disk (Render free etc.) set the
    PROFILES_JSON env var instead — a JSON string with the same shape; it
    wins over the file and survives server restarts.
    """
    env_profiles = os.environ.get("PROFILES_JSON", "").strip()
    if env_profiles:
        try:
            data = json.loads(env_profiles)
            return {k: v for k, v in data.items()
                    if isinstance(k, str) and isinstance(v, str)}
        except json.JSONDecodeError:
            print("WARNING: PROFILES_JSON env var is not valid JSON — ignored")
    if not os.path.exists(PROFILES_PATH):
        template = {
            "default": (
                "Name: Your Name\n"
                "Email: you@example.com\n"
                "City: Your City\n"
                "Interests: ...\n"
                "Edit this file to describe whoever the forms should be filled for."
            )
        }
        try:
            with open(PROFILES_PATH, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
        except OSError:
            return {}
    try:
        with open(PROFILES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def profile_text(name: str) -> str:
    profiles = load_profiles()
    return profiles.get(name) or profiles.get("default", "")


def _entry_overrides(request: Request):
    """entry.* query params pasted into the magic URL are forwarded to the
    final Google Form URL, overriding the AI's answers for those entries."""
    return [(k, v) for k, v in request.query_params.multi_items()
            if k.startswith("entry.")]


def _search_flag(request: Request) -> bool | None:
    """?search=0/1 per request; None -> use the WEB_SEARCH env default."""
    v = request.query_params.get("search", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return None


def compute_fill(form_id: str, profile_name: str, entry_overrides=None,
                  use_search=None) -> dict:
    """Detect the form, ask Gemini, build the pre-filled URL."""
    view_url = viewform_url_for(form_id)
    try:
        title, questions = parse_form(view_url)
    except FormParseError as e:
        raise HTTPException(422, str(e))
    profile = profile_text(profile_name)
    if "Your Name" in profile:
        print("WARNING: profile is still the auto-created template — set the "
              "PROFILES_JSON env var (or edit profiles.json) so forms fill "
              "with real data.")
    try:
        if use_search is None:
            use_search = WEB_SEARCH
        answers = ai.generate_answers(gemini_client(), profile, title,
                                      questions, use_search=use_search)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()          # full reason in the uvicorn log
        raise HTTPException(502, f"Gemini call failed: {e}")
    fill_url, applied, skipped_required = build_prefill(
        view_url, questions, answers, entry_overrides)
    return {
        "title": title,
        "fill_url": fill_url,
        "applied": applied,
        "applied_count": len(applied),
        "skipped_required": skipped_required,
        "template_profile": "Your Name" in profile,
    }


def cached_fill(form_id: str, profile_name: str, entry_overrides=None,
                use_search=None) -> dict:
    overrides = tuple(entry_overrides or ())
    key = (form_id, profile_name, overrides, use_search)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    if len(_cache) > 128:
        _cache.clear()
    result = compute_fill(form_id, profile_name, list(overrides), use_search)
    _cache[key] = (time.time(), result)
    return result


# --------------------------------------------------------------------------
# keyword / profile resolution
# --------------------------------------------------------------------------

def _url_keyword(raw: str) -> str | None:
    """Keyword embedded in the pasted URL:
    docs.<kw>.google.com/...  or  <kw>.google.com/..."""
    host = raw.split("/", 1)[0].lower()
    if host.endswith(".google.com"):
        labels = host.split(".")
        if len(labels) == 3 and labels[0] != "docs":
            return labels[0]
        if len(labels) >= 4:
            return labels[1]
    return None


def _host_keyword(request: Request) -> str | None:
    """Keyword from the Host header — works when you host the app on a
    wildcard domain (https://lakshay.your-domain.com/forms/...)."""
    host = (request.headers.get("host") or "").split(":")[0].lower().strip()
    if not host or host == "localhost" or re.fullmatch(r"[\d.]+", host):
        return None
    label = host.split(".")[0]
    if label in ("www", "docs", "forms", "app", "api", "formfill"):
        return None
    return label


def _pick_profile(request: Request, raw: str) -> str:
    """Profile precedence: ?p= > keyword in the URL > Host-header keyword > default."""
    explicit = request.query_params.get("p")
    if explicit:
        return explicit
    profiles = load_profiles()
    for cand in (_url_keyword(raw), _host_keyword(request)):
        if cand and cand in profiles:
            return cand
    return "default"


# --------------------------------------------------------------------------
# HTML pages (only used for errors and the optional ?preview=1)
# --------------------------------------------------------------------------

def _error_page(e: HTTPException):
    msg = html_mod.escape(str(e.detail))
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>FormFill</title><style>body{margin:0;font-family:system-ui,sans-serif;"
        "background:#0f1420;color:#e8edf7;display:flex;align-items:center;"
        "justify-content:center;min-height:100vh;text-align:center}"
        ".box{max-width:520px;padding:24px}h2{font-size:18px}"
        ".err{color:#ff7a7a;white-space:pre-wrap;font-size:14px}"
        "button{margin-top:18px;background:#5b8cff;color:#fff;border:0;"
        "border-radius:8px;padding:12px 22px;font-size:15px;cursor:pointer}"
        "</style></head><body><div class='box'><h2>Couldn't fill this form</h2>"
        f"<p class='err'>{msg}</p>"
        "<button onclick='location.reload()'>Try again</button>"
        "</div></body></html>",
        status_code=e.status_code,
    )


def _preview_page(result: dict):
    def esc(s):
        return html_mod.escape(str(s))

    rows = "".join(
        f"<div class='r'><b>{esc(a['label'])}</b>{esc(a['answer'])}</div>"
        for a in result["applied"]
    ) or "<p class='dim'>(no answers were applied)</p>"
    template_warn = ""
    if result.get("template_profile"):
        template_warn = ("<p class='warn'>profiles.json on the server still has the "
                         "auto-created template profile — edit it so forms fill with "
                         "real data.</p>")
    skipped = ""
    if result["skipped_required"]:
        names = ", ".join(esc(q["label"]) for q in result["skipped_required"])
        skipped = f"<p class='warn'>Required but unanswered: {names}</p>"
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='robots' content='noindex'><title>FormFill</title><style>"
        "body{margin:0;font-family:system-ui,sans-serif;background:#0f1420;"
        "color:#e8edf7;display:flex;align-items:center;justify-content:center;"
        "min-height:100vh}.box{max-width:560px;padding:24px;width:100%}"
        "h2{font-size:18px;margin:0 0 6px}.dim{color:#8b98b3;font-size:14px}"
        ".ans{margin-top:14px;text-align:left;font-size:14px;border-top:1px solid #2a3651;"
        "padding-top:10px;max-height:45vh;overflow-y:auto}"
        ".r{padding:6px 0;border-bottom:1px solid #1c2438}"
        ".r b{color:#8b98b3;font-weight:500;display:block;font-size:12px}"
        ".warn{margin-top:14px;color:#e8c46a;font-size:13px}"
        "a.btn{display:inline-block;margin-top:18px;background:#5b8cff;color:#fff;"
        "border-radius:8px;padding:12px 22px;font-size:15px;text-decoration:none}"
        "</style></head><body><div class='box'>"
        f"<h2>{esc(result['title'] or 'Form')}</h2>"
        f"<p class='dim'>{result['applied_count']} answer(s) ready</p>"
        f"<div class='ans'>{rows}</div>{template_warn}{skipped}"
        f"<a class='btn' href='{esc(result['fill_url'])}'>Open the filled form</a>"
        "</div></body></html>"
    )


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class ParseRequest(BaseModel):
    url: str


class PrefillRequest(BaseModel):
    url: str
    profile: str


class MagicRequest(BaseModel):
    url: str


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/api/profiles")
def api_profiles(x_access_code: str | None = Header(default=None)):
    check_access(header=x_access_code)
    return {"names": list(load_profiles().keys())}


@app.post("/api/magiclink")
def api_magiclink(req: MagicRequest, x_access_code: str | None = Header(default=None)):
    """Convert any Google Form URL (or forms.gle link) into a form id."""
    check_access(header=x_access_code)
    form_id = form_id_from_url(req.url)
    if not form_id:
        raise HTTPException(422, "That doesn't look like a Google Form URL.")
    return {"form_id": form_id, "magic_path": f"/f/{form_id}"}


@app.post("/api/parse")
def api_parse(req: ParseRequest, x_access_code: str | None = Header(default=None)):
    check_access(header=x_access_code)
    try:
        form_title, questions = parse_form(req.url)
    except FormParseError as e:
        raise HTTPException(422, str(e))
    return {
        "title": form_title,
        "question_count": len(questions),
        "questions": [q.to_dict() for q in questions],
    }


@app.post("/api/prefill")
def api_prefill(req: PrefillRequest, x_access_code: str | None = Header(default=None)):
    check_access(header=x_access_code)
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Server has no GEMINI_API_KEY configured.")
    try:
        title, questions = parse_form(req.url)
    except FormParseError as e:
        raise HTTPException(422, str(e))
    try:
        answers = ai.generate_answers(gemini_client(), req.profile, title, questions)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, f"Gemini call failed: {e}")
    fill_url, applied, skipped_required = build_prefill(req.url, questions, answers)
    return {
        "title": title,
        "fill_url": fill_url,
        "applied": applied,
        "applied_count": len(applied),
        "skipped_required": skipped_required,
    }


@app.get("/api/fill/{form_id}")
def api_fill(request: Request, form_id: str, p: str = "default",
             code: str | None = None,
             x_access_code: str | None = Header(default=None)):
    """JSON API: everything the magic link does, as data."""
    check_access(header=x_access_code, query=code)
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Server has no GEMINI_API_KEY configured.")
    return cached_fill(form_id, p, _entry_overrides(request),
                      _search_flag(request))


# --------------------------------------------------------------------------
# magic links — server-side fill, direct redirect (no loading page)
# --------------------------------------------------------------------------

@app.get("/f/{form_id}")
def magic_redirect(request: Request, form_id: str):
    code = request.query_params.get("code")
    check_access(query=code)
    p = request.query_params.get("p")
    if p is None:
        hk = _host_keyword(request)
        profiles = load_profiles()
        p = hk if hk in profiles else "default"
    try:
        result = cached_fill(form_id, p, _entry_overrides(request),
                             _search_flag(request))
    except HTTPException as e:
        return _error_page(e)
    if request.query_params.get("preview") == "1":
        return _preview_page(result)
    return RedirectResponse(result["fill_url"], status_code=302)


@app.get("/{target:path}")
def magic_url(request: Request, target: str):
    """Paste-after-host magic URL. Accepts the pasted URL with or without
    scheme, with any host prefix (docs.<keyword>.google.com, <keyword>.google.com)
    or a leading <keyword>/ path prefix. All work happens server-side and the
    browser is redirected straight to the filled form. ?preview=1 shows a
    review page first. The keyword selects the profile from profiles.json."""
    raw = (target or "").strip().lstrip("/")
    if raw.lower().startswith(("http://", "https://")):
        raw = raw.split("://", 1)[1]
    # optional path-prefix keyword: <keyword>/rest-of-url
    if "/" in raw:
        first, _, rest = raw.partition("/")
        if ("." not in first and len(first) > 1
                and not first.lower().startswith(("http", "docs", "forms", "www"))):
            candidate = first
            rest = rest.lstrip("/")
            profiles = load_profiles()
            raw = rest
            if candidate in profiles:
                request.state.kw = candidate
    if not form_id_from_url(raw) and not raw.lower().startswith(("docs.google.com", "forms.gle")):
        kw = getattr(request.state, "kw", None)
        if not kw and _url_keyword(raw) is None:
            raise HTTPException(404, "Not found. Paste a Google Form URL right after "
                                      "the server address, e.g. http://server/https://docs.google.com/forms/d/e/.../viewform")
    code = request.query_params.get("code")
    check_access(query=code)
    form_id = form_id_from_url(raw)
    if not form_id:
        raise HTTPException(422, "Could not find a Google Form id in that URL.")
    p = request.query_params.get("p")
    if p is None:
        profiles = load_profiles()
        p = "default"
        for cand in (getattr(request.state, "kw", None), _url_keyword(raw),
                     _host_keyword(request)):
            if cand and cand in profiles:
                p = cand
                break
    try:
        result = cached_fill(form_id, p, _entry_overrides(request),
                             _search_flag(request))
    except HTTPException as e:
        return _error_page(e)
    if request.query_params.get("preview") == "1":
        return _preview_page(result)
    return RedirectResponse(result["fill_url"], status_code=302)


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
