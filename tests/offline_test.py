"""Offline tests: parser against live public forms, prefill with mock answers,
and the FastAPI endpoints with a mocked Gemini call. No API key needed.

Run from the project root:  python tests/offline_test.py
"""

import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from formparser import FormParseError, parse_form, extract_fb_data, parse_questions
from prefill import build_prefill, match_option

LIVE_FORMS = [
    "https://docs.google.com/forms/d/e/1FAIpQLSfAPTqB6KZhrS0B15Y-_M2WtBcuumL1RK6VH4JN95_Bk0rYiw/viewform",
    "https://docs.google.com/forms/d/e/1FAIpQLSdOdg9EZn5TsRGzNCFJIxRm277PSjyMddowNmPBI9361T9N2w/viewform",
]

PASS = 0
def ok(msg):
    global PASS
    PASS += 1
    print(f"  ok - {msg}")


def test_parse_live():
    print("parser: live forms")
    for url in LIVE_FORMS:
        title, qs = parse_form(url)
        assert qs, f"no questions parsed from {url}"
        types = {q.qtype for q in qs}
        print(f"    {title!r}: {len(qs)} questions, types={sorted(types)}")
        ok(f"parsed {len(qs)} questions from {url.split('/e/')[1][:12]}…")
    # Every question has required index/entry_id/label
    _, qs = parse_form(LIVE_FORMS[0])
    for q in qs:
        assert isinstance(q.index, int) and isinstance(q.entry_id, int) and q.label
    ok("all questions have index, entry_id, label")
    # Grid rows use the row label
    grids = [q for q in qs if q.qtype == "grid"]
    if grids:
        assert any("SQD" in q.label or q.options for q in grids)
        ok(f"grid rows parsed with row labels (e.g. {grids[0].label[:40]!r})")


def test_parse_errors():
    print("parser: error handling")
    for bad in ["", "not-a-url", "https://example.com/"]:
        try:
            parse_form(bad)
            assert False, f"should have failed: {bad!r}"
        except FormParseError:
            ok(f"rejected {bad!r}")


def test_prefill():
    print("prefill: builder")
    # Fabricate a question set covering every supported type
    class Q:
        def __init__(self, **kw): self.__dict__.update(kw)
        def to_dict(self): return {"index": self.index, "label": self.label, "type": self.qtype}
    qs = [
        Q(index=0, qtype="short_answer", entry_id=111, label="Name", required=True),
        Q(index=1, qtype="paragraph", entry_id=222, label="Comments", required=False),
        Q(index=2, qtype="multiple_choice", entry_id=333, label="Gender",
          required=True, options=["Male", "Female", "Other"]),
        Q(index=3, qtype="checkboxes", entry_id=444, label="Interests",
          required=False, options=["Music", "Sports", "Reading"]),
        Q(index=4, qtype="linear_scale", entry_id=555, label="Rating",
          required=False, options=None, scale_min=1, scale_max=5),
        Q(index=5, qtype="date", entry_id=666, label="Birthday",
          required=False, options=None, scale_min=None, scale_max=None),
        Q(index=6, qtype="time", entry_id=777, label="Start time",
          required=False, options=None, scale_min=None, scale_max=None),
        Q(index=7, qtype="grid", entry_id=888, label="I am satisfied",
          required=False, options=["Strongly Agree", "Agree", "Disagree"]),
    ]
    answers = {
        0: "Vishal Verma",
        1: "Great service, thanks & <sorry> for the delay",   # needs encoding
        2: "male",                                            # fuzzy match
        3: "Music, reading, Cooking",                         # Cooking invalid -> dropped
        4: "7",                                               # out of range -> dropped
        5: "1999-08-19",
        6: "14:30",
        7: "strongly agree",                                  # fuzzy match
    }
    url, applied, skipped = build_prefill(
        "https://docs.google.com/forms/d/e/TEST/viewform?usp=sharing", qs, answers)
    base, _, query = url.partition("?")
    assert base == "https://docs.google.com/forms/d/e/TEST/viewform", base
    assert query.startswith("usp=pp_url&"), query
    p = dict(urllib.parse.parse_qsl(query))
    assert p["entry.111"] == "Vishal Verma"
    assert p["entry.333"] == "Male", p
    assert "entry.555" not in p, "out-of-range scale should be dropped"
    assert p["entry.666_year"] == "1999" and p["entry.666_month"] == "8" and p["entry.666_day"] == "19"
    assert p["entry.777_hour"] == "14" and p["entry.777_minute"] == "30"
    assert p["entry.888"] == "Strongly Agree"
    # '&' and '<' properly encoded
    assert "entry.222" in p and "%26" in query or p.get("entry.222", "").find("&") == -1
    assert "sorry" in p["entry.222"]
    # checkbox: only valid options applied
    vals = [v for k, v in urllib.parse.parse_qsl(query) if k == "entry.444"]
    assert sorted(vals) == ["Music", "Reading"], vals
    # name question (index 0) was answered so not in skipped even though required
    ok("encoding, fuzzy matching, date/time, checkbox filtering all correct")
    assert skipped == [], skipped
    # missing required answer reported
    url2, applied2, skipped2 = build_prefill("https://x/viewform", qs, {2: "Other"})
    assert any(s["index"] == 0 for s in skipped2), skipped2
    ok("unanswered required questions are reported")


def test_match_option():
    print("prefill: option matching")
    opts = ["New Delhi", "Mumbai", "Chennai"]
    assert match_option("new delhi", opts) == "New Delhi"
    assert match_option("I live in Mumbai city", opts) == "Mumbai"
    assert match_option("Kolkata", opts) is None
    assert match_option("", opts) is None
    ok("exact / casefold / substring / miss")


def test_api(monkeypatch_ai=True):
    print("api: endpoints (Gemini mocked)")
    import app as appmod
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    c = TestClient(appmod.app)

    r = c.post("/api/parse", json={"url": "not-a-url"})
    assert r.status_code == 422, r.text
    ok("bad URL -> 422")

    r = c.post("/api/parse", json={"url": LIVE_FORMS[0]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["question_count"] == len(d["questions"]) > 0
    ok(f"parse endpoint: {d['question_count']} questions, title={d['title'][:30]!r}")

    fake_answers = {}
    for q in d["questions"]:
        if q["type"] in ("short_answer", "paragraph"):
            fake_answers[q["index"]] = "Test Person"
        elif q["options"]:
            fake_answers[q["index"]] = q["options"][0]
        elif q["type"] == "date":
            fake_answers[q["index"]] = "2024-05-01"
        elif q["type"] == "time":
            fake_answers[q["index"]] = "10:00"
        elif q["type"] == "linear_scale":
            fake_answers[q["index"]] = str(q["scale_min"] or 1)

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kw):
                class R: pass
                r = R()
                r.parsed = None
                r.text = json.dumps([{"index": i, "answer": a} for i, a in fake_answers.items()])
                return r
    appmod.GEMINI_API_KEY = "test-key"
    with patch.object(appmod.ai, "generate_answers", return_value=fake_answers):
        r = c.post("/api/prefill", json={"url": LIVE_FORMS[0], "profile": "Name: Test Person"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["fill_url"].startswith("https://docs.google.com/forms/d/e/1FAIpQLSfAPT")
    assert "usp=pp_url&" in d["fill_url"]
    assert d["applied_count"] > 0
    print(f"    prefill: {d['applied_count']}/{d['question_count'] if 'question_count' in d else ''} applied, url len={len(d['fill_url'])}")
    ok("prefill endpoint returns encoded pp_url")
    # ACCESS_CODE gate
    appmod.ACCESS_CODE = "team-secret"
    r = c.post("/api/parse", json={"url": LIVE_FORMS[0]})
    assert r.status_code == 401
    r = c.post("/api/parse", json={"url": LIVE_FORMS[0]}, headers={"X-Access-Code": "team-secret"})
    assert r.status_code == 200
    appmod.ACCESS_CODE = ""
    ok("access-code gate works (401 without, 200 with)")


# ---------------------------------------------------------------- magic link
from formparser import form_id_from_url, viewform_url_for

def test_url_helpers():
    print("links: url -> form id")
    u = "https://docs.google.com/forms/d/e/1FAIpQLSc55zJe2eHL3zczn6BGK9zG44bfCBByuCY9M6uDBEsCJC7rLA/viewform"
    assert form_id_from_url(u) == "1FAIpQLSc55zJe2eHL3zczn6BGK9zG44bfCBByuCY9M6uDBEsCJC7rLA"
    assert form_id_from_url("1FAIpQLSabc123_XY") == "1FAIpQLSabc123_XY"
    assert form_id_from_url("https://docs.google.com/forms/d/1xyzABC456/edit") == "1xyzABC456"
    assert form_id_from_url("https://random.site/page") is None
    assert form_id_from_url("") is None
    assert viewform_url_for("1FAIpQLSabc") == "https://docs.google.com/forms/d/e/1FAIpQLSabc/viewform"
    assert viewform_url_for("1xyz") == "https://docs.google.com/forms/d/1xyz/viewform"
    ok("id extraction + viewform url building")

def test_magic_link(monkeypatch_ai=True):
    print("api: magic link")
    import app as appmod
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    c = TestClient(appmod.app)

    cnr = TestClient(appmod.app, follow_redirects=False)
    fid = form_id_from_url(LIVE_FORMS[0])
    appmod.GEMINI_API_KEY = "test-key"
    _, _qs = parse_form(LIVE_FORMS[0])
    _fake = {q.index: (q.options[0] if q.options else "X") for q in _qs}
    with patch.object(appmod.ai, "generate_answers", return_value=_fake):
        r = cnr.get(f"/f/{fid}?p=default")
    assert r.status_code == 302, r.status_code
    assert r.headers["location"].startswith("https://docs.google.com/forms/d/e/"), r.headers["location"]
    assert "usp=pp_url&" in r.headers["location"]
    ok("GET /f/<id> -> direct 302 redirect to filled form (no loader)")

    # /api/magiclink resolves a docs.google.com URL
    r = c.post("/api/magiclink", json={"url": LIVE_FORMS[0]})
    assert r.status_code == 200 and r.json()["form_id"] == fid
    r = c.post("/api/magiclink", json={"url": "https://example.com/x"})
    assert r.status_code == 422
    ok("POST /api/magiclink (resolve + reject)")

    # /api/profiles lists names incl. default
    r = c.get("/api/profiles")
    assert r.status_code == 200 and "default" in r.json()["names"]
    ok("GET /api/profiles")

    # preview page is server-rendered (optional ?preview=1)
    with patch.object(appmod.ai, "generate_answers", return_value={0: "X"}):
        appmod._cache.clear()
        rp = cnr.get(f"/f/{fid}?p=default&preview=1")
    assert rp.status_code == 200 and "answer(s) ready" in rp.text, rp.status_code
    assert "docs.google.com" in rp.text
    ok("?preview=1 shows server-rendered review page")

    # /api/fill with mocked Gemini
    _, qs = parse_form(LIVE_FORMS[0])
    fake = {}
    for q in qs:
        if q.options: fake[q.index] = q.options[0]
        elif q.qtype in ("short_answer", "paragraph"): fake[q.index] = "Magic Test"
    appmod.GEMINI_API_KEY = "test-key"
    with patch.object(appmod.ai, "generate_answers", return_value=fake):
        r = c.get(f"/api/fill/{fid}?p=default")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["fill_url"].startswith("https://docs.google.com/forms/d/e/")
    assert "usp=pp_url&" in d["fill_url"] and d["applied_count"] > 0
    # second call is served from cache and still works
    with patch.object(appmod.ai, "generate_answers", side_effect=AssertionError("should be cached")):
        r2 = c.get(f"/api/fill/{fid}?p=default")
    assert r2.status_code == 200 and r2.json()["fill_url"] == d["fill_url"]
    ok("GET /api/fill/<id> fills + caches")

    # access code via query param (?code=) for browser-opened links
    appmod.ACCESS_CODE = "team-secret"
    r = cnr.get(f"/f/{fid}")
    assert r.status_code == 401
    with patch.object(appmod.ai, "generate_answers", return_value=fake):
        r = cnr.get(f"/f/{fid}?code=team-secret")
    assert r.status_code == 302
    r = c.get(f"/api/fill/{fid}?code=wrong")
    assert r.status_code == 401
    appmod.ACCESS_CODE = ""
    ok("?code= gate works for magic links")


# ------------------------------------------------------------- model + catch-all
import ai as aimod

class FakeModel:
    def __init__(self, name): self.name = name

class FakeModels:
    def __init__(self, names): self._names = ["models/" + n for n in names]
    def list(self): return [FakeModel(n) for n in self._names]

class FakeClient:
    def __init__(self, names): self.models = FakeModels(names)

def test_model_resolution():
    print("ai: model resolution")
    # configured model available -> used as-is
    aimod._resolved_model = None; aimod._available = None
    c = FakeClient(["gemini-3.6-flash", "gemini-2.0-flash-lite"])
    assert aimod.resolve_model(c) == "gemini-3.6-flash"
    # configured model retired -> newest stable flash picked
    aimod._resolved_model = None; aimod._available = None
    c = FakeClient(["gemini-2.0-flash", "gemini-4.1-flash", "gemini-4.1-flash-exp", "gemini-3.5-flash-lite"])
    assert aimod.resolve_model(c) == "gemini-4.1-flash", aimod.resolve_model(c)
    # listing fails -> configured default survives
    aimod._resolved_model = None; aimod._available = None
    class BrokenList:
        def list(self): raise RuntimeError("no listing")
    class BrokenClient:
        models = BrokenList()
    assert aimod.resolve_model(BrokenClient()) == aimod.MODEL
    aimod._resolved_model = None
    ok("configured model, retirement fallback, listing failure fallback")

def test_catchall_route():
    print("api: paste-after-host URL")
    import app as appmod
    from fastapi.testclient import TestClient
    c = TestClient(appmod.app)
    cnr = TestClient(appmod.app, follow_redirects=False)
    u = LIVE_FORMS[0]  # https://docs.google.com/forms/d/e/<id>/viewform
    inner = u[len("https://"):]
    r = cnr.get("/" + u)                       # with scheme -> direct redirect
    assert r.status_code == 302 and r.headers["location"].startswith("https://docs.google.com"), r.status_code
    r2 = cnr.get("/" + inner)                  # without scheme
    assert r2.status_code == 302
    r3 = cnr.get("/" + u + "?preview=1")       # explicit preview
    assert r3.status_code == 200 and "answer(s) ready" in r3.text
    r4 = cnr.get("/favicon.ico")               # not a form URL -> normal 404
    assert r4.status_code == 404
    r5 = cnr.get("/https://example.com/whatever")
    assert r5.status_code == 404
    # access code gate still applies
    appmod.ACCESS_CODE = "s3cret"
    r6 = cnr.get("/" + u)
    assert r6.status_code == 401
    r7 = cnr.get("/" + u + "?code=s3cret")
    assert r7.status_code == 302
    appmod.ACCESS_CODE = ""
    ok("scheme/without-scheme/preview/404s/access gate + direct redirects")


# ------------------------------------------------------------- retries / 503
class FakeErr(Exception):
    def __init__(self, code): super().__init__(f"http {code}"); self.status_code = code

class Resp:
    parsed = None
    text = '[{"index": 0, "answer": "recovered"}]'

def _flaky_client(names, fail_times, code=503):
    calls = {"n": 0}
    class M:
        def list(self): return [type("NM", (), {"name": "models/" + n})() for n in names]
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] <= fail_times:
                raise FakeErr(code)
            return Resp()
    class C: models = M()
    return C(), calls

def test_retry_logic():
    print("ai: retry / fallback logic")
    import time as _time
    from unittest.mock import patch
    # 1. transient 503s then success -> retries and recovers
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    c, calls = _flaky_client(["gemini-3.6-flash"], fail_times=2)
    with patch("time.sleep") as _s:
        out = aimod.generate_answers(c, "p", "t", [])
    assert out == {0: "recovered"}, out
    assert calls["n"] == 3 and _s.call_count == 2
    ok("503 x2 then success: retried with backoff, recovered")
    # 2. non-transient 400: schema attempt then plain attempt, then stop (no model 2)
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()
    c, calls = _flaky_client(["gemini-3.6-flash"], fail_times=99, code=400)
    try:
        aimod.generate_answers(c, "p", "t", [], use_search=False)
        assert False, "should raise"
    except FakeErr:
        pass
    assert calls["n"] == 3, calls["n"]   # schema + json + plain configs only
    ok("400: schema->plain fallback, no pointless retries")
    # 3. persistent 503 on primary -> falls back to lite model
    aimod._resolved_model = None; aimod._available = None
    calls = {"n": 0}
    class M2:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"}),
                                type("NM", (), {"name": "models/gemini-3.6-flash-lite"})]
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            calls.setdefault("models", []).append(model)
            if model == "gemini-3.6-flash":
                raise FakeErr(503)
            return Resp()
    class C2: models = M2()
    with patch("time.sleep"):
        out = aimod.generate_answers(C2(), "p", "t", [])
    assert out == {0: "recovered"}
    assert "gemini-3.6-flash-lite" in calls["models"]
    ok("persistent 503: fell back to flash-lite model")
    aimod._resolved_model = None; aimod._available = None

    # 4. model has no JSON mode -> falls through to the plain-text config
    class NoJsonErr(Exception):
        def __init__(self):
            super().__init__("400 JSON mode is not enabled for this model")
            self.status_code = 400
    class NoJsonModels:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash-lite"})]
        def generate_content(self, model, contents, config):
            if getattr(config, "response_mime_type", None):
                raise NoJsonErr()
            class PlainResp:
                parsed = None
                text = "```json\n[{\"index\": 0, \"answer\": \"plain-mode ok\"}]\n```"
            return PlainResp()
    class NoJsonClient: models = NoJsonModels()
    aimod._resolved_model = None; aimod._available = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()
    with patch("time.sleep"):
        out = aimod.generate_answers(NoJsonClient(), "p", "t", [], use_search=False)
    assert out == {0: "plain-mode ok"}, out
    aimod._resolved_model = None; aimod._available = None
    ok("no-JSON-mode model: plain-text config + fence stripping works")



# ------------------------------------------------------- keyword -> profile
import tempfile, os as _os

def test_keyword_profiles():
    print("api: custom keyword URLs pick the profile")
    import app as appmod
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    cnr = TestClient(appmod.app, follow_redirects=False)
    fid = form_id_from_url(LIVE_FORMS[0])

    # temp profiles.json with a lakshay profile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"default": "Name: Default", "lakshay": "Name: Lakshay"}, tmp)
    tmp.close()
    old_path = appmod.PROFILES_PATH
    appmod.PROFILES_PATH = tmp.name
    appmod.GEMINI_API_KEY = "test-key"

    seen = {}
    def fake_gen(client, profile, title, questions, use_search=None):
        seen["profile"] = profile
        return {q.index: (q.options[0] if q.options else "X") for q in questions}

    try:
        # 1. docs.<keyword>.google.com host style
        appmod._cache.clear()
        with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
            r = cnr.get(f"/docs.lakshay.google.com/forms/d/e/{fid}/viewform")
        assert r.status_code == 302 and seen["profile"] == "Name: Lakshay", seen
        # 2. <keyword>/ prefix style
        appmod._cache.clear()
        with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
            r = cnr.get(f"/lakshay/https://docs.google.com/forms/d/e/{fid}/viewform")
        assert r.status_code == 302 and seen["profile"] == "Name: Lakshay", seen
        # 3. no keyword -> default
        appmod._cache.clear()
        with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
            r = cnr.get(f"/https://docs.google.com/forms/d/e/{fid}/viewform")
        assert r.status_code == 302 and seen["profile"] == "Name: Default", seen
        # 4. unknown keyword -> default (never crashes)
        appmod._cache.clear()
        with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
            r = cnr.get(f"/docs.someoneelse.google.com/forms/d/e/{fid}/viewform")
        assert r.status_code == 302 and seen["profile"] == "Name: Default", seen
        # 5. ?p= beats the keyword
        appmod._cache.clear()
        with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
            r = cnr.get(f"/docs.lakshay.google.com/forms/d/e/{fid}/viewform?p=default")
        assert r.status_code == 302 and seen["profile"] == "Name: Default", seen
    finally:
        appmod.PROFILES_PATH = old_path
        _os.unlink(tmp.name)
    ok("docs.<kw>.google.com / <kw>/ prefix / default / unknown / ?p override")


# --------------------------------------------- quota / model-choice behaviour
class QuotaErr(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 429

def test_quota_and_model_choice():
    print("ai: 429 retry-delay + non-text model filtering")
    from unittest.mock import patch
    # 1. 'Please retry in Xs' hint is honoured
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    calls = {"n": 0}
    class QM:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise QuotaErr("...Please retry in 27.5s. ...")
            return Resp()
    class QC: models = QM()
    with patch("time.sleep") as _sl:
        out = aimod.generate_answers(QC(), "p", "t", [])
    assert out == {0: "recovered"}, out
    assert _sl.call_count == 1 and abs(_sl.call_args[0][0] - 27.5) < 0.01, _sl.call_args_list
    ok("429 retry-delay hint parsed and honoured")
    # 2. TTS / image models are never auto-picked
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    c = FakeClient(["gemini-3.8-flash-lite-tts", "gemini-2.0-flash"])
    assert aimod.resolve_model(c) == "gemini-2.0-flash", aimod.resolve_model(c)
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    class LiteList:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.8-flash-lite-tts"}),
                                type("NM", (), {"name": "models/gemini-3.6-flash-lite"})]
    class LiteClient: models = LiteList()
    assert aimod._auto_lite(LiteClient()) == "gemini-3.6-flash-lite", aimod._auto_lite(LiteClient())
    ok("tts/image models excluded from auto-selection")
    # 3. last working model is remembered and tried first
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    order = []
    class WM:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"}),
                                type("NM", (), {"name": "models/gemini-3.6-flash-lite"})]
        def generate_content(self, model, contents, config):
            order.append(model)
            if model == "gemini-3.6-flash":
                raise FakeErr(503)      # permanent overload on primary
            return Resp()
    class WMC: models = WM()
    with patch("time.sleep"):
        aimod.generate_answers(WMC(), "p", "t", [])
    assert order[0] == "gemini-3.6-flash" and order[-1] == "gemini-3.6-flash-lite"
    # next call: lite model first (no wasted 503 round on primary)
    order.clear()
    with patch("time.sleep"):
        aimod.generate_answers(WMC(), "p", "t", [])
    assert order[0] == "gemini-3.6-flash-lite", order
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    ok("last working model remembered and tried first")


# ------------------------------------------- entry params injected via URL
def test_entry_param_forwarding():
    print("api: entry.* params in the magic URL are forwarded")
    import app as appmod
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    cnr = TestClient(appmod.app, follow_redirects=False)
    fid = form_id_from_url(LIVE_FORMS[0])
    _, qs = parse_form(LIVE_FORMS[0])
    fake = {q.index: (q.options[0] if q.options else "AI answer") for q in qs}
    appmod.GEMINI_API_KEY = "test-key"
    # user supplies an entry param that overrides one AI answer
    some_entry = f"entry.{qs[0].entry_id}"
    appmod._cache.clear()
    with patch.object(appmod.ai, "generate_answers", return_value=fake):
        r = cnr.get(f"/https://docs.google.com/forms/d/e/{fid}/viewform"
                    f"?{some_entry}=My+Custom+Name")
    assert r.status_code == 302
    loc = r.headers["location"]
    assert f"{some_entry}=My+Custom+Name" in loc, loc
    assert "usp=pp_url&" in loc
    # the overridden entry appears exactly once (no duplicate from the AI)
    assert loc.count(f"{some_entry}=") == 1, loc
    # AI answers for other questions are still present
    assert "entry." in loc
    ok("entry.* params forwarded and overriding AI answers")


# ------------------------------------------------ web-search grounding
def test_search_grounding():
    print("ai: Google-Search grounding + fallback")
    from unittest.mock import patch
    import google.genai.types as gtypes

    # 1. configs: search on -> ONE plain-text config with the tool (frugal)
    aimod._no_search_models.clear(); aimod._no_json_models.clear(); aimod._last_good = None
    cfgs = aimod._build_configs(gtypes, True, "gemini-3.6-flash")
    assert len(cfgs) == 1, cfgs
    assert getattr(cfgs[0][1], "tools", None), cfgs
    assert getattr(cfgs[0][1], "response_mime_type", None) is None  # plain text
    cfgs_off = aimod._build_configs(gtypes, False, "gemini-3.6-flash")
    assert len(cfgs_off) == 3 and not any(getattr(c, "tools", None) for _, c in cfgs_off)
    assert [k for k, _ in cfgs_off] == ["schema", "json", "plain"]
    ok("config chain: 1 plain+search when searching, schema->json->plain otherwise")

    # 2. API rejects search+schema -> retries without search and succeeds
    class SearchErr(Exception):
        def __init__(self):
            super().__init__("400 Search grounding is not supported with tools here")
            self.status_code = 400
    seen_tools = []
    class SM:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, model, contents, config):
            has_tools = bool(getattr(config, "tools", None))
            seen_tools.append(has_tools)
            if has_tools:
                raise SearchErr()
            return Resp()
    class SC: models = SM()
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    with patch("time.sleep"):
        out = aimod.generate_answers(SC(), "p", "t", [])
    assert out == {0: "recovered"}, out
    assert seen_tools[0] is True and seen_tools[-1] is False, seen_tools
    ok("search-unsupported -> clean retry without search")

    # 3. ?search=0 flows from the magic URL into generate_answers
    import app as appmod
    from fastapi.testclient import TestClient
    cnr = TestClient(appmod.app, follow_redirects=False)
    fid = form_id_from_url(LIVE_FORMS[0])
    captured = {}
    def fake_gen(client, profile, title, questions, use_search=None):
        captured["use_search"] = use_search
        return {q.index: (q.options[0] if q.options else "X") for q in questions}
    appmod.GEMINI_API_KEY = "test-key"
    appmod._cache.clear()
    with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
        r = cnr.get(f"/https://docs.google.com/forms/d/e/{fid}/viewform?search=0")
    assert r.status_code == 302 and captured["use_search"] is False, captured
    appmod._cache.clear()
    with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
        r = cnr.get(f"/https://docs.google.com/forms/d/e/{fid}/viewform")
    assert r.status_code == 302 and captured["use_search"] is True, captured  # env default on
    ok("?search=0/1 per-link flag reaches the AI call")
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None


# --------------------------------- frugal quota use + OpenAI-compatible API
def test_one_call_per_fill():
    print("ai: warm fill costs exactly one API request")
    from unittest.mock import patch
    calls = {"n": 0}
    class M:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            return Resp()
    class C: models = M()
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()
    aimod.generate_answers(C(), "p", "t", [])           # cold fill
    cold = calls["n"]
    aimod.generate_answers(C(), "p", "t", [])           # warm fill
    assert calls["n"] == cold + 1, (cold, calls["n"])
    ok(f"cold fill {cold} request(s), warm fill exactly 1")

def test_no_search_sticky():
    print("ai: search-rejecting model never gets search again")
    from unittest.mock import patch
    class SearchErr(Exception):
        def __init__(self):
            super().__init__("400 Search grounding is not supported with tools")
            self.status_code = 400
    tools_seen = []
    class M:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, model, contents, config):
            tools_seen.append(bool(getattr(config, "tools", None)))
            if getattr(config, "tools", None):
                raise SearchErr()
            return Resp()
    class C: models = M()
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()
    aimod.generate_answers(C(), "p", "t", [])
    n_after_first = len(tools_seen)
    aimod.generate_answers(C(), "p", "t", [])
    assert tools_seen[:1] == [True] and tools_seen[-1] is False
    assert not any(tools_seen[n_after_first:]), tools_seen
    ok("after the first rejection, search is skipped for that model")
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()

def test_openai_compat_client():
    print("ai: OpenAI-compatible provider (Groq/OpenRouter style)")
    from unittest.mock import patch
    client = aimod.OpenAICompatClient("https://api.example.com/v1/", "test-key")

    class FakeResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"index": 0, "answer": "groq ok"}]'}}]}
    class FakeErr:
        status_code = 429
        text = '{"error": {"message": "rate limited, retry in 12.5s"}}'
        def json(self):
            return {"error": {"message": "rate limited, retry in 12.5s"}}
    sent = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, payload=json)
        return FakeResp()
    with patch("requests.post", side_effect=fake_post):
        out = aimod.generate_answers(client, "p", "t", [])
    assert out == {0: "groq ok"}, out
    assert sent["url"] == "https://api.example.com/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["payload"]["model"] == aimod.MODEL
    assert sent["payload"]["response_format"] == {"type": "json_object"}
    assert "messages" in sent["payload"]
    ok("compat client maps configs to chat.completions + parses answers")
    # error path: HTTP 429 -> retryable with parsed delay
    with patch("requests.post", side_effect=[FakeErr(), FakeResp()]):
        with patch("time.sleep") as _sl:
            out = aimod.generate_answers(client, "p", "t", [])
    assert out == {0: "groq ok"}
    assert _sl.call_count == 1 and abs(_sl.call_args[0][0] - 12.5) < 0.01, _sl.call_args_list
    ok("compat client: 429 with 'retry in Xs' honoured like Gemini")


# ------------------------------------------------ multi-key rotation
def _rot_fake(statuses=None):
    """A fake genai-like client; raises FakeErr(statuses[i]) on the i-th
    call before succeeding."""
    class M:
        def __init__(self):
            self.calls = 0
        def list(self):
            return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, **kw):
            self.calls += 1
            if statuses and self.calls <= len(statuses):
                raise FakeErr(statuses[self.calls - 1])
            return Resp()
    class C:
        def __init__(self):
            self.models = M()
    return C()

def test_key_rotation():
    print("ai: multi-key rotation")
    from unittest.mock import patch
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()

    # 1. round-robin: successes spread across keys
    a, b = _rot_fake(), _rot_fake()
    rot = aimod.RotatingClient([a, b])
    for _ in range(4):
        aimod.generate_answers(rot, "p", "t", [])
    assert a.models.calls == 2 and b.models.calls == 2, (a.models.calls, b.models.calls)
    ok("requests round-robin evenly across keys")

    # 2. 429 on the current key -> instantly retried on the next key, no sleep
    a, b = _rot_fake(), _rot_fake()
    class Always429:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, **kw): raise FakeErr(429)
    class A:
        models = Always429()
    rot = aimod.RotatingClient([A(), b])
    with patch("time.sleep") as _sl:
        out = aimod.generate_answers(rot, "p", "t", [])
    assert out == {0: "recovered"}, out
    assert b.models.calls == 1 and _sl.call_count == 0, (_sl.call_args_list, b.models.calls)
    ok("429 on one key -> immediate retry on the next key (no waiting)")

    # 3. non-retryable 400: no key rotation wasted
    class Always400:
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, **kw): raise FakeErr(400)
    class B:
        models = Always400()
    a, b = _rot_fake(), _rot_fake()
    rot = aimod.RotatingClient([B(), a])
    try:
        aimod.generate_answers(rot, "p", "t", [])
        assert False, "should raise"
    except FakeErr:
        pass
    assert a.models.calls == 0, a.models.calls
    ok("400 raises immediately — other keys not wasted")

    # 4. key-list parsing
    import app as appmod
    assert appmod._parse_keys(" k1 , k2 ,,k3\nk4 ") == ["k1", "k2", "k3", "k4"]
    assert appmod._parse_keys("") == []
    ok("comma/newline key-list parsing")
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()


# ---------------------------------------- third-party gateway pool
def test_compat_endpoints_pool():
    print("app: COMPAT_ENDPOINTS + mixed provider pool")
    import os as _os
    from unittest.mock import patch
    import app as appmod

    # 1. parsing: named / unnamed / malformed / empty
    eps = appmod._parse_compat_endpoints(
        "codecraft|https://codecraftapi.com/v1|k1|m1;tokenbargain|https://tokenbargain.dev/v1|k2|m2")
    assert eps == [("codecraft", "https://codecraftapi.com/v1", "k1", "m1"),
                   ("tokenbargain", "https://tokenbargain.dev/v1", "k2", "m2")], eps
    assert appmod._parse_compat_endpoints("https://x/v1|k|m") == [("compat", "https://x/v1", "k", "m")]
    assert appmod._parse_compat_endpoints("") == []
    assert appmod._parse_compat_endpoints("bad-entry") == []
    ok("COMPAT_ENDPOINTS parsing (named / unnamed / malformed)")

    # 2. app builds a rotating pool from Gemini keys + compat endpoints
    saved = {k: _os.environ.get(k) for k in
             ("GEMINI_API_KEYS", "COMPAT_ENDPOINTS", "AI_PROVIDER")}
    saved_client = appmod._gemini_client
    saved_key_const = appmod.GEMINI_API_KEY
    try:
        appmod.GEMINI_API_KEY = "k1,k2"
        _os.environ.pop("GEMINI_API_KEYS", None)
        _os.environ.pop("AI_PROVIDER", None)
        _os.environ["COMPAT_ENDPOINTS"] = (
            "codecraft|https://codecraftapi.com/v1|k3|m3;"
            "tokenbargain|https://tokenbargain.dev/v1|k4|m4")
        appmod._gemini_client = None
        c = appmod.gemini_client()
        assert isinstance(c, aimod.RotatingClient) and c.size == 4, c.size
        # entries carry their own models for the gateways
        assert [m for _, m in c._entries] == [None, None, "m3", "m4"]
    finally:
        for k, v in saved.items():
            if v is None: _os.environ.pop(k, None)
            else: _os.environ[k] = v
        appmod._gemini_client = saved_client
        appmod.GEMINI_API_KEY = saved_key_const
    ok("pool = 2 Gemini keys + 2 gateways, per-endpoint models")

    # 3. mixed-pool failover: Gemini entry 429 -> gateway serves with its model
    class Rec:
        def __init__(self, fail=None):
            self.fail = fail; self.seen = []; self.models = _RM(self)
    class _RM:
        def __init__(self, rec): self.rec = rec
        def list(self): return [type("NM", (), {"name": "models/gemini-3.6-flash"})]
        def generate_content(self, model, contents, config):
            self.rec.seen.append(model)
            if self.rec.fail: raise FakeErr(self.rec.fail)
            return Resp()
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()
    gem = Rec(fail=429)      # gemini key out of quota
    cc = Rec()               # third-party gateway works
    pool = aimod.RotatingClient([(gem, None), (cc, "cc-gpt-model")])
    with patch("time.sleep") as _sl:
        out = aimod.generate_answers(pool, "p", "t", [])
    assert out == {0: "recovered"}, out
    assert gem.seen == ["gemini-3.6-flash"], gem.seen     # passed-through model
    assert cc.seen == ["cc-gpt-model"], cc.seen           # endpoint's own model wins
    assert _sl.call_count == 0, _sl.call_args_list        # instant switch, no wait
    ok("gateway model used; 429 hops providers without sleeping")
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()


# ------------------------------------ robust answer-shape parsing
def test_answer_shapes():
    print("ai: robust parsing of model answer shapes")
    import json as _json
    tm = aimod._to_map
    # 1. the shape we ask for
    assert tm([{"index": 0, "answer": "a"}, {"index": 3, "answer": "b"}]) == {0: "a", 3: "b"}
    # 2. wrapper object
    assert tm({"answers": [{"index": 1, "answer": "x"}]}) == {1: "x"}
    assert tm({"result": [{"index": 1, "answer": "x"}]}) == {1: "x"}
    # 3. index-keyed object
    assert tm({"0": "a", "2": "b"}) == {0: "a", 2: "b"}
    # 4. bare positional array
    assert tm(["a", "b", "c"]) == {0: "a", 1: "b", 2: "c"}
    # 5. pairs
    assert tm([[0, "a"], [5, "b"]]) == {0: "a", 5: "b"}
    # 6. alternate keys + missing index falls back to position
    assert tm([{"question": 2, "value": "a"}, {"answer": "b"}]) == {2: "a", 1: "b"}
    # 7. index as string
    assert tm([{"index": "4", "answer": "a"}]) == {4: "a"}
    # 8. junk entries are skipped, not fatal
    assert tm([{"index": 0, "answer": "a"}, {"weird": True}]) == {0: "a"}
    # 9. _loads: fenced and prose-wrapped JSON
    assert aimod._loads('```json\n[{"index":0,"answer":"a"}]\n```') == [{"index": 0, "answer": "a"}]
    assert aimod._loads('Sure! Here you go:\n[{"index":0,"answer":"a"}]\nHope that helps.') == [{"index": 0, "answer": "a"}]
    ok("9 answer shapes parsed (wrapper/keyed/positional/pairs/alts/junk/fences)")

def test_gateway_odd_shape_end_to_end():
    print("ai: gateway returning wrapped object still fills the form")
    from unittest.mock import patch
    # simulates what CodeCraft-style gateways can return: object, no index field
    class OddM:
        def list(self): return []
        def generate_content(self, model, contents, config):
            class R:
                parsed = None
                text = '{"answers": [{"question": 0, "answer": "Name: Test"}, {"question": 1, "answer": "10th"}]}'
            return R()
    class OddC: models = OddM()
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()
    with patch("time.sleep"):
        out = aimod.generate_answers(OddC(), "p", "t", [], use_search=False)
    assert out == {0: "Name: Test", 1: "10th"}, out
    ok("wrapped/no-index response parsed into correct answers")
    aimod._resolved_model = None; aimod._available = None; aimod._working_model = None
    aimod._last_good = None; aimod._no_json_models.clear(); aimod._no_search_models.clear()


# ---------------------------- single-object answers + answer-everything prompt
def test_single_object_and_prompt():
    print("ai: single-object answers + answer-everything prompt")
    tm = aimod._to_map
    # the exact shape the CodeCraft gateway returned
    assert tm({"question_index": 0, "answer": "Your Name"}) == {0: "Your Name"}, \
        tm({"question_index": 0, "answer": "Your Name"})
    assert tm({"question": 5, "value": "42"}) == {5: "42"}
    # wrappers and index-keyed still work (regression)
    assert tm({"answers": [{"index": 1, "answer": "x"}]}) == {1: "x"}
    assert tm({"0": "a", "2": "b"}) == {0: "a", 2: "b"}
    ok("single answer object parsed; wrappers/keyed unaffected")

    # the prompt now demands every question answered
    from formparser import parse_form
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    import app as appmod
    prompt_text = {}
    def fake_gen(client, profile, title, questions, use_search=None):
        prompt_text["p"] = aimod.build_prompt(profile, title, questions)
        return {q.index: "ans" for q in questions}
    cnr = TestClient(appmod.app, follow_redirects=False)
    appmod.GEMINI_API_KEY = "test-key"
    appmod._cache.clear()
    with patch.object(appmod.ai, "generate_answers", side_effect=fake_gen):
        cnr.get(f"/f/{form_id_from_url(LIVE_FORMS[0])}")
    p = prompt_text["p"]
    assert "Answer EVERY question" in p and "NOT" in p and "personal" in p
    assert "omit only" in p
    ok("prompt: maths/GK explicitly not personal; only unknown personal qs omitted")


# ---------------------------- PROFILES_JSON env (hosting-friendly profiles)
def test_profiles_json_env():
    import app as appmod
    print("profiles: PROFILES_JSON env var support")
    os.environ["PROFILES_JSON"] = json.dumps(
        {"default": "Name: Vishal Verma\nEmail: v@example.com",
         "lakshay": "Name: Lakshay"})
    try:
        assert appmod.profile_text("default") == \
            "Name: Vishal Verma\nEmail: v@example.com"
        assert appmod.profile_text("lakshay") == "Name: Lakshay"
        assert appmod.profile_text("nope") == \
            "Name: Vishal Verma\nEmail: v@example.com"   # falls back
        os.environ["PROFILES_JSON"] = "{broken json"
        assert appmod.load_profiles() in ({}, {"default": ""}) or \
            isinstance(appmod.load_profiles(), dict)
    finally:
        os.environ.pop("PROFILES_JSON", None)
    ok("PROFILES_JSON wins over file, bad JSON ignored safely")


if __name__ == "__main__":
    test_parse_live()
    test_parse_errors()
    test_prefill()
    test_match_option()
    test_api()
    test_url_helpers()
    test_magic_link()
    test_model_resolution()
    test_catchall_route()
    test_retry_logic()
    test_keyword_profiles()
    test_quota_and_model_choice()
    test_entry_param_forwarding()
    test_search_grounding()
    test_one_call_per_fill()
    test_no_search_sticky()
    test_openai_compat_client()
    test_key_rotation()
    test_compat_endpoints_pool()
    test_answer_shapes()
    test_gateway_odd_shape_end_to_end()
    test_single_object_and_prompt()
    test_profiles_json_env()
    print(f"\nALL {PASS} CHECKS PASSED")
