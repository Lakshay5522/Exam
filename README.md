# FormFill

Self-hosted tool that uses **Google Gemini** to pre-fill **Google Forms**.
Turn any Google Form URL into a link on *your* server that auto-fills the form.

## The magic link (main feature)

Take a Google Form URL:

```
https://docs.google.com/forms/d/e/1FAIpQLSc55zJe2eHL3zczn6BGK9zG44bfCBByuCY9M6uDBEsCJC7rLA/viewform
```

And paste it right after your server's address (two supported formats):

```
http://your-server:8000/https://docs.google.com/forms/d/e/1FAIpQLSc55zJe.../viewform
http://your-server:8000/f/1FAIpQLSc55zJe...
```

All formats now do the work **entirely in the backend**: the server
detects the form, asks Gemini, and returns a **direct redirect** to the
filled form — no loading page, no preview step. (First open takes 3-10s
while Gemini answers; repeat opens are served from a 10-minute cache.)

Options on the magic link:

### Custom keyword = the person to fill as

Put a keyword (a profile name from `profiles.json`) into the URL:

```
http://your-server:8000/docs.lakshay.google.com/forms/d/e/<id>/viewform
http://your-server:8000/lakshay/https://docs.google.com/forms/d/e/<id>/viewform
https://lakshay.your-domain.com/forms/...          (when hosted on a wildcard domain)
```

Each of these fills the form as the `lakshay` profile. A keyword that
doesn't match a profile falls back to `default`.

| Query | Effect |
|---|---|
| `?p=<name>` | use a named profile from `profiles.json` (overrides any keyword) |
| `?entry.<id>=<value>` | **inject an answer directly into the URL** — forwarded to the final Google Form and overrides the AI's answer for that entry (repeatable for checkboxes) |
| `?search=0` | disable Google-Search grounding for this link (on by default) |
| `&preview=1` | show a review page with the answers before redirecting |
| `&code=<access code>` | required if `ACCESS_CODE` is set in `.env` |

Example: fill everything with AI but force your own name —

```
http://server/https://docs.google.com/forms/d/e/<id>/viewform?entry.1000026=Lakshay
```

The index page also has a converter box that builds this link for you, and a
manual mode (paste URL + profile → pre-filled link) if you prefer that.

## Multiple free API keys (no investment needed)

Every Google account can create its own free Gemini key at
https://aistudio.google.com/apikey — no billing required. Put several in
`.env`, comma-separated:

```ini
GEMINI_API_KEYS=key-from-account-1,key-from-account-2,key-from-account-3
```

The server then round-robins every request across the keys, and when one
key hits its free-tier quota (429), the very next attempt instantly uses
the next key — no waiting. Three keys ≈ three times the fills per minute,
and combined with the one-request-per-warm-fill behaviour that's plenty
for personal use.

Keep it to a few keys from your own accounts: creating many accounts
just to multiply quota goes against Google's terms and risks the keys
being banned.

## Third-party token gateways (codecraftapi, tokenbargain, ...)

Sites that hand out free monthly token budgets via OpenAI-compatible
endpoints work too, and they rotate TOGETHER with your Gemini keys in one
pool — a quota error on any one instantly moves the request to the next.
In `.env` (entries separated by `;`, fields by `|`):

```ini
COMPAT_ENDPOINTS=codecraft|https://codecraftapi.com/v1|cc-your-key|their-model-name;tokenbargain|https://tokenbargain.dev/v1|tb-your-key|their-model-name
```

Copy the exact **base URL** and **model name** from each site's own
dashboard/docs page (the two fields the gateway needs). Any
OpenAI-compatible gateway works with this format (Groq, OpenRouter,
Cerebras, local Ollama...).

Two honest caveats:
- These gateways relay your traffic — the form's questions and your
  profile text pass through their servers. Fine for school quizzes;
  don't route anything sensitive through them.
- Their reliability is not Google's. The rotation handles outages
  automatically (5xx = instant next provider), but expect occasional
  slow or weird responses from smaller relays.

## Alternative AI providers

The default is Google Gemini (best free quality, web-search grounding).
If you keep hitting free-tier rate limits, FormFill works with **any
OpenAI-compatible provider** — in `.env`:

```ini
AI_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

Popular free options: **Groq** (very fast, high request limits — no web
search), **OpenRouter** (free `:free` models with daily caps — includes
many model families), **Cerebras**, **Together**, or a local **Ollama**
(`OPENAI_BASE_URL=http://localhost:11434/v1`). Retry, quota-hint and
fallback logic all work against these too.

Honest comparison for form-filling on free tiers:

| Provider | Free limits | Quality | Web search |
|---|---|---|---|
| Gemini (default) | a few requests/min per model | best | yes (grounding) |
| Groq | much higher RPM | very good | no |
| OpenRouter free models | daily caps | varies by model | no |

Quota frugality: after the first successful fill, the server remembers
the exact (model, search, output-mode) combination that worked and reuses
it — a warm fill costs **exactly one API request**. Results are cached 30
minutes per (form, profile), so repeat opens cost nothing.

## Web-search questions

Grounding with **Google Search** is ON by default: while answering, the
model can look things up on the live web — current affairs, GK, "latest
X" questions, anything past its training cutoff. Maths and textbook
questions keep working as before (the model computes those itself).

- Per link: add `?search=0` to skip search (faster, fewer quota hits).
- Server-wide: set `WEB_SEARCH=0` in `.env`.
- If a model/API combination rejects the search tool, the fill retries
  once without it — a fill never fails because of search.
- Grounded requests use a separate, smaller daily quota on the free tier
  and add a few seconds per fill.

## Profiles

Answers come from the profile text in `profiles.json` next to `app.py`
(auto-created on first run — edit it!):

```json
{
  "default": "Name: Vishal Verma\nEmail: vishal@example.com\nCity: Delhi\nInterests: cricket, music",
  "friend": "Name: ...\nEmail: ..."
}
```

Then `https://your-server/f/<form-id>?p=friend` fills as "friend".

## Setup

```bash
cd formfill
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your Gemini key
uvicorn app:app --host 0.0.0.0 --port 8000
```

Free Gemini key: https://aistudio.google.com/apikey

Deploying for the team: results are cached for 10 minutes per (form,
profile) so repeat opens are instant and don't burn Gemini quota.

## Hosting it

**Option A — Render (free, always-on URL).** Push this folder to a GitHub
repo, create a *Blueprint* on https://render.com, select the repo (the
included `render.yaml` is picked up automatically) and set `GEMINI_API_KEY`
when prompted. You get `https://your-app.onrender.com` — then
`https://your-app.onrender.com/f/<form-id>` works anywhere.

**Option B — expose your phone/PC instantly (Cloudflare Tunnel).** Keep
running it on Termux/localhost, then:

```bash
cloudflared tunnel --url http://localhost:8000
```

You immediately get a public `https://xyz.trycloudflare.com` URL to share.

**Option C — own domain (wildcard subdomains = profiles).** Host the app
(Option A or any VPS), point `*.your-domain.com` at it, and every profile
gets its own subdomain: `https://lakshay.your-domain.com/forms/...`
The app reads the subdomain from the Host header automatically.

## Troubleshooting

- **"Gemini call failed: 400 ... JSON mode is not enabled"** — the model
  being used (often the flash-lite fallback) doesn't support JSON-mode
  responses. The server now falls through to a plain-text request and
  parses the JSON out of the reply, so this should no longer surface.
- **"Gemini call failed: 'index'"** — the model/gateway returned answers
  in a different JSON shape than expected. The parser now accepts every
  reasonable shape (wrapped objects, index-keyed maps, bare positional
  arrays, alternate field names, prose around the JSON), and if something
  is still unparseable the raw response is printed in the server log so
  the format can be reported and supported.
- **"Gemini call failed: 503 ... high demand"** — Google's model was
  temporarily overloaded; nothing is wrong with your setup. The server now
  retries automatically (waits 2s/4s/8s), and if the model stays overloaded
  it falls back to a lighter flash-lite model. The preview page also gets a
  "Try again" button so you can retry manually.
- **"Gemini call failed: 429 ... quota" (free tier)** — the free tier allows
  only a handful of requests per minute per model. The server reads
  Google's own "Please retry in Xs" hint and waits exactly that long, then
  retries; it also remembers the last model that worked and tries it
  first, and never auto-picks non-text models (tts/image variants).
  Results are cached 10 minutes per (form, profile), so normal use stays
  well inside the limits. For heavy shared use, enable billing on your
  Google AI Studio project.
- **"Gemini call failed: 404 ... no longer available"** — the configured
  model was retired by Google. The default is `gemini-3.6-flash`, and if
  that (or any `GEMINI_MODEL` you set) isn't available to your account, the
  server automatically picks the newest flash model your account CAN see.
- **"Gemini call failed: ..." (other)** — the real reason is in the red status
  text in the UI and in the uvicorn terminal (full traceback). The most
  common cause is an invalid API key (create a fresh one in Google AI Studio
  and paste it into `.env` with no spaces/quotes). The code automatically
  retries without the response schema if your SDK version rejects it.
- **"No form data found" (422)** — the form link must be a *published* Google
  Form (openable in an incognito browser), and `forms.gle` short links are
  resolved automatically at link-creation time.

## How it works

1. `formparser.py` downloads the form page and extracts the embedded
   `FB_PUBLIC_LOAD_DATA_` JSON with a **balanced JSON decoder** (not a regex),
   mapping it to typed questions (entry id, type, options, required flag).
   Grid rows are handled correctly (row label at index 3, columns as options).
2. `ai.py` sends numbered questions + the profile to Gemini (default
   `gemini-3.6-flash`) with **structured output** (`response_schema`): every
   answer is tagged with its question index, so answers can never shift onto
   the wrong question. Transient 429/5xx errors are retried with backoff, an
   overloaded model falls back to flash-lite, retired model names are
   auto-replaced, and markdown-fence stripping is built in.
3. `prefill.py` **validates every answer against the question type** — choice
   answers must match a real option (fuzzy-matched), scales are range-checked,
   dates/times parsed — then builds the pre-fill URL with proper encoding
   (`?usp=pp_url&entry.<id>=...`). Unanswered *required* questions are flagged.
4. `app.py` exposes the magic link (`/f/<id>` → loader → `/api/fill/<id>` →
   redirect), the JSON API, profiles, and a small response cache.

## Test without a Gemini key

```bash
python tests/offline_test.py
```

49 checks: live-form parsing, model fallback, paste-URL routing, 503 retry/fallback, keyword profiles, direct redirects, prefill validation, URL-id extraction, magic-link
loader, caching, and access-code gating (Gemini mocked).

## Notes & limits

- **Multi-page forms work.** The prefill URL carries answers for every page:
  page 1 opens filled, and clicking Next shows page 2 already filled — no
  reload, nothing gets stuck. (Verified against a 22-section government
  survey: all 26 answers across all pages load.)
- Only **published** Google Forms work (any form openable while logged out).
- File-upload/image questions are skipped (not pre-fillable via URL).
- `FB_PUBLIC_LOAD_DATA_` is a Google internal — if it ever changes,
  `formparser.py` is the only file to update.
- This generates links that humans review and submit; it never submits a form
  by itself. Use in line with Google's Terms of Service.


## Free 24/7 hosting (phone-only, no payment)

FormFill is a Python backend, so static hosts (GitHub Pages, plain Netlify)
cannot run it. The free option that works and needs no card is **Render**
(blueprint included — `render.yaml`):

1. **GitHub** — make an account if needed, create a new *empty* repo
   (e.g. `formfill`, keep it **private** so your code isn't public).
   In Termux:
   ```
   cd /storage/emulated/0/Download/formfill
   git init && git add . && git commit -m "formfill"
   git remote add origin https://github.com/<you>/formfill.git
   git push -u origin main
   ```
   (Termux's git will open the browser once to log you in.)
   **Do not commit `.env`** — it is git-ignored by the included
   `.gitignore`; on Render you type keys into the dashboard instead.
2. **Render** — sign up with GitHub at render.com, then
   New + → *Blueprint* → pick the repo. It reads `render.yaml` and asks
   for your secrets: paste your key into `GEMINI_API_KEY` (or use the
   CodeCraft gateway: `AI_PROVIDER=openai`,
   `OPENAI_BASE_URL=https://codecraftapi.com/v1`,
   `OPENAI_MODEL=gemini-3.7-flash`, `OPENAI_API_KEY=...`). Deploy.
3. **Your profile** — free servers reset their disk on restart, so
   instead of editing `profiles.json` on the server, set the
   `PROFILES_JSON` env var in Render (Environment tab) to the same JSON
   you have in the file:
   `{"default": "Name: ...\nEmail: ..."}` — it survives restarts.
4. **Keep it awake** — free Render services sleep after ~15 min idle
   (first request then takes ~30–60 s). Fix it free: make an
   uptimerobot.com account, add an HTTP monitor for
   `https://your-app.onrender.com/` every 10 minutes. Then it is
   effectively 24/7.

Your magic URL from anywhere becomes:
`https://formfill-xxxx.onrender.com/f/<form-id>`

Other free hosts: **Hugging Face Spaces** (Docker SDK, sleeps after 48 h,
restarts on first hit) and **Koyeb** (one small free instance) both work
with the same uvicorn command. **Vercel/Netlify** are serverless —
10–60 s function limits and no persistent cache — not recommended for
the AI call chain. If you later want your own domain with keyword URLs
(`docs.lakshay.<domain>`), a Cloudflare tunnel or a Cloudflare-proxied
domain in front of Render works (see the tunnel notes above).
