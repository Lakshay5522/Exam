"""Parse a published Google Form into a structured question list.

Instead of scraping the rendered HTML, this reads the FB_PUBLIC_LOAD_DATA_
JSON blob that Google embeds in every published form page. Extraction uses
a balanced JSON decoder (raw_decode) rather than a regex capture, so
braces or semicolons inside question text can never truncate the payload.
"""

import json
import re

import requests

# Google's internal section-type codes -> our names
TYPE_NAMES = {
    0: "short_answer",
    1: "paragraph",
    2: "multiple_choice",
    3: "dropdown",
    4: "checkboxes",
    5: "linear_scale",
    7: "grid",
    9: "date",
    10: "time",
}

CHOICE_TYPES = {"multiple_choice", "dropdown", "checkboxes", "grid"}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


class FormParseError(Exception):
    """Raised when a URL is not a readable published Google Form."""


class Question:
    __slots__ = (
        "index", "entry_id", "qtype", "label", "section_title",
        "required", "options", "scale_min", "scale_max",
    )

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    def to_dict(self):
        return {
            "index": self.index,
            "type": self.qtype,
            "label": self.label,
            "section": self.section_title,
            "required": self.required,
            "options": self.options,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }


def fetch_form_html(url: str, timeout: int = 25) -> str:
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        raise FormParseError("URL must start with http:// or https://")
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.RequestException as e:
        raise FormParseError(f"Could not reach the form: {e}") from e
    if resp.status_code != 200:
        raise FormParseError(
            f"Form returned HTTP {resp.status_code} — is the link correct and public?"
        )
    return resp.text


def extract_fb_data(html: str):
    m = re.search(r"FB_PUBLIC_LOAD_DATA_\s*=\s*", html)
    if not m:
        raise FormParseError(
            "No form data found — the URL is probably not a published Google Form."
        )
    try:
        # Balanced parse: never truncates on braces/semicolons inside text.
        data, _ = json.JSONDecoder().raw_decode(html, m.end())
    except json.JSONDecodeError as e:
        raise FormParseError(f"Form data is malformed: {e}") from e
    return data


def _text(v):
    return v.strip() if isinstance(v, str) else ""


def parse_questions(data) -> tuple[str, list[Question]]:
    """Convert the FB blob into (form_title, flat list of questions)."""
    try:
        sections = data[1][1]
        form_title = _text(data[3]) if len(data) > 3 and data[3] else ""
    except (TypeError, IndexError) as e:
        raise FormParseError("Unexpected form data structure.") from e

    questions: list[Question] = []
    for section in sections:
        if not section or len(section) < 5 or not section[4]:
            continue
        code = section[3]
        qtype = TYPE_NAMES.get(code)
        if qtype is None:  # titles, images, file upload etc. — not answerable
            continue
        section_title = _text(section[1])
        for item in section[4]:
            if not item or not isinstance(item[0], int):
                continue
            label = section_title
            options = None
            scale_min = scale_max = None
            if qtype == "grid":
                # Each item is one ROW: [entry_id, columns, required, [row_label], ...]
                if len(item) > 3 and item[3]:
                    label = _text(item[3][0]) or section_title
                options = [o[0] for o in (item[1] or []) if o and o[0]]
            elif qtype == "linear_scale":
                options = [str(o[0]) for o in (item[1] or []) if o and o[0]]
                if len(item) > 3 and item[3] and len(item[3]) >= 2:
                    scale_min, scale_max = item[3][0], item[3][1]
            elif qtype in CHOICE_TYPES:
                options = [o[0] for o in (item[1] or []) if o and o[0]]
            questions.append(Question(
                index=len(questions), entry_id=item[0], qtype=qtype,
                label=label, section_title=section_title,
                required=bool(item[2]) if len(item) > 2 else False,
                options=options, scale_min=scale_min, scale_max=scale_max,
            ))
    if not questions:
        raise FormParseError("No answerable questions found in this form.")
    return form_title, questions


def parse_form(url: str) -> tuple[str, list[Question]]:
    html = fetch_form_html(url)
    data = extract_fb_data(html)
    return parse_questions(data)


# ---------------------------------------------------------------------------
# URL <-> form-id helpers (for magic links)
# ---------------------------------------------------------------------------

def form_id_from_url(url_or_id: str) -> str | None:
    """Accept a bare form id, a docs.google.com URL, or a forms.gle short
    link (resolved via redirect). Returns the form id or None."""
    s = (url_or_id or "").strip()
    if not s:
        return None
    if re.fullmatch(r"1FAIpQLS[\w-]+", s):
        return s
    if "forms.gle/" in s:
        try:
            r = requests.get(s, headers=_HEADERS, timeout=25)
            s = r.url  # after redirects
        except requests.RequestException:
            return None
    m = (re.search(r"/forms/d/e/([\w-]+)/", s)
         or re.search(r"/forms/d/([\w-]+)/", s)
         or re.search(r"/forms/u/\d+/d/e/([\w-]+)/", s))
    return m.group(1) if m else None


def viewform_url_for(form_id: str) -> str:
    """Canonical public viewform URL for a form id."""
    if form_id.startswith("1FAIpQLS"):
        return f"https://docs.google.com/forms/d/e/{form_id}/viewform"
    return f"https://docs.google.com/forms/d/{form_id}/viewform"
