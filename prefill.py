"""Validate AI answers per question type and build the pre-filled URL."""

from datetime import datetime
from urllib.parse import urlencode

PREFILL_FLAG = "usp=pp_url"


def match_option(value: str, options):
    """Match an AI free-text answer to a real option. Returns the exact
    option text, or None if nothing matches well enough."""
    if not options:
        return None
    value = value.strip()
    if not value:
        return None
    low = value.casefold()
    for opt in options:                      # exact
        if opt.casefold() == low:
            return opt
    for opt in options:                      # option contained in answer
        if opt.casefold() in low:
            return opt
    for opt in options:                      # answer contained in option
        if low in opt.casefold():
            return opt
    return None


def _apply(question, value: str):
    """Return a list of (param_name, param_value) for one question,
    or None if the answer is unusable for this question type."""
    v = value.strip()
    if not v or v.lower() in ("none", "n/a", "unknown"):
        return None
    qtype = question.qtype
    eid = f"entry.{question.entry_id}"

    if qtype in ("short_answer", "paragraph"):
        return [(eid, v)]

    if qtype in ("multiple_choice", "dropdown", "grid"):
        m = match_option(v, question.options or [])
        return [(eid, m)] if m else None

    if qtype == "checkboxes":
        picked = []
        for part in v.split(","):
            m = match_option(part, question.options or [])
            if m and m not in picked:
                picked.append(m)
        return [(eid, p) for p in picked] if picked else None

    if qtype == "linear_scale":
        try:
            n = int(float(v))
        except ValueError:
            return None
        lo = question.scale_min if question.scale_min is not None else 1
        hi = question.scale_max if question.scale_max is not None else 5
        if lo <= n <= hi:
            return [(eid, str(n))]
        return None

    if qtype == "date":
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                d = datetime.strptime(v, fmt)
            except ValueError:
                continue
            return [
                (f"{eid}_year", str(d.year)),
                (f"{eid}_month", str(d.month)),
                (f"{eid}_day", str(d.day)),
            ]
        return None

    if qtype == "time":
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p"):
            try:
                t = datetime.strptime(v, fmt)
            except ValueError:
                continue
            return [
                (f"{eid}_hour", str(t.hour)),
                (f"{eid}_minute", str(t.minute)),
            ]
        return None

    return None


def build_prefill(form_url: str, questions, answers_by_index: dict,
                  entry_overrides=None):
    """answers_by_index: {question.index: raw AI answer string}.

    entry_overrides: optional list of (param, value) pairs like
    ("entry.1000026", "My Name") taken from the magic URL's query string —
    injected verbatim into the final URL, overriding the AI's answer for
    the same entry.

    Returns (prefill_url, applied, skipped_required) where `applied` is a
    list of {"index", "label", "answer"} dicts actually written into the
    URL and `skipped_required` lists required questions left unanswered.
    """
    params, applied, answered = [], [], set()
    for q in questions:
        raw = answers_by_index.get(q.index)
        if raw is None:
            continue
        pairs = _apply(q, raw)
        if pairs:
            params.extend(pairs)
            display = ", ".join(dict.fromkeys(pv for _, pv in pairs))
            applied.append({
                "index": q.index, "label": q.label, "answer": display,
            })
            answered.add(q.index)
    skipped_required = [
        q.to_dict() for q in questions if q.required and q.index not in answered
    ]
    if entry_overrides:
        override_keys = {k for k, _ in entry_overrides}
        params = [(k, v) for k, v in params if k not in override_keys]
        params.extend(entry_overrides)
        for k, v in entry_overrides:
            applied.append({"index": -1, "label": f"{k} (from URL)", "answer": v})

    base = form_url.split("?", 1)[0]
    if not params:
        return base + "?" + PREFILL_FLAG, applied, skipped_required
    return f"{base}?{PREFILL_FLAG}&{urlencode(params)}", applied, skipped_required
