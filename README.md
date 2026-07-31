# Intelligent Agent-Based Detection of Malicious HTTP Logs

Python backend that analyzes batches of web server access logs, detects anomalous
activity and decides the response through two orchestrated LLM agents.

---

## Objective

Given a batch of HTTP access logs, answer two questions: is there a threat here,
and what should be done about it.

Detection does not use a rule list. A character language model is fitted on benign
traffic only, so anything that does not look like the site's normal traffic scores
high — including attack families the system has never seen. The dataset labels are
used to evaluate, never to train.

The pipeline is a funnel. A deterministic layer scores every event in
milliseconds and reduces thousands of them to a handful of incident dossiers; only
those dossiers reach the LLM. A triage agent classifies each incident, and an
action agent decides the proportional response — block, rate limit, alert, monitor
or allow.

Calibration is recall-first: an attack classified as benign is the one error the
system must not make, while a false positive only creates triage work, and triage is
exactly what the agents do. Of the 362 five-minute windows containing an attack in
the held-out set, 361 reach an agent.

Confirmed incidents are reported to Telegram. The agents run on Anthropic or Gemini,
and the service still works with no key at all.

<!-- architecture diagram -->

---

## How to use

### Run with Docker

```bash
docker build -t log-detector .
docker run -p 8000:8000 --env-file .env log-detector
```

`--env-file` is what carries the LLM key and the Telegram token into the container;
without it the service still runs, just on the deterministic path with alerts going
to the log. `docker compose up` reads `.env` on its own and keeps the database in a
volume across restarts.

### Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

The service starts with no configuration required. A trained model ships in
`models/`, blocking runs in dry-run mode, and without an LLM key the agents are
replaced by a deterministic fallback.

### Check that it is up

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "llm_enabled": false, "block_mode": "dry_run"}
```

### Analyze a batch

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"events": [
        {"ip": "203.0.113.9", "time": "2019-01-23 03:00:00+00:00",
         "url": "/image/1?wh=50x50%27%20UNION%20ALL%20SELECT%20NULL--",
         "status": 200, "user_agent": "sqlmap/1.4"}
      ]}'
```

Raw Combined Log Format lines are accepted in the same array:

```json
{"events": ["203.0.113.5 - - [23/Jan/2019:03:00:00 +0000] \"GET /wp-login.php HTTP/1.1\" 404 0 \"-\" \"python-requests/2.21.0\""]}
```

The response reports the threat verdict, the suggested action, and for each
incident the events that triggered it together with the exact substring the model
found improbable.

### Enable the LLM agents

Either provider works. Whichever key is present is picked automatically.

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or
export GEMINI_API_KEY=AIza...

uvicorn app.main:app --port 8000
```

Defaults are `claude-opus-5` and `gemini-3.5-flash`; set `LLM_MODEL` to pick another,
and `LLM_PROVIDER=anthropic|gemini` to force one when both keys exist. Confirm which
one is live:

```bash
curl -s http://localhost:8000/health | jq '.detail | {llm_provider, llm_model}'
```

Without any key everything still works; the verdicts just come from the deterministic
path and are marked `analyzed_by: deterministic`.

The agent loop is written against one client surface, so a provider is an adapter and
not a rewrite — see `app/agents/providers.py`. On free-tier Gemini quotas expect the
occasional `429`; a provider failure degrades that incident to the deterministic
verdict and the request still returns `200`.

### Report incidents to Telegram

Use your own bot. Talk to [@BotFather](https://t.me/BotFather), send `/newbot` and
keep the token it gives you. Then send any message to your new bot and read your chat
id from `https://api.telegram.org/bot<TOKEN>/getUpdates` — it is the
`result[0].message.chat.id` field.

```bash
export TELEGRAM_BOT_TOKEN=<your-token>
export TELEGRAM_CHAT_ID=<your-chat-id>
uvicorn app.main:app --port 8000
```

Any incident whose severity is `medium` or higher now sends a message. This one was
delivered by a real run — the body below the header is the action agent's own
wording, not a template:

<img width="523" height="342" alt="image" src="https://github.com/user-attachments/assets/8bbc9d38-0976-484b-bbb7-f0ce3fc35a97" />

The detector never saw the string `ThinkPHP`, has no notion of a framework version
and cannot tell a 200 from a 404. All of that — the vulnerable component, the
payload's intent, the fact that a 200 with a body suggests the command ran, and the
instruction to go look under `/tmp/.x` — comes from the agent reading the dossier.

Notification is wired at two points: the deterministic path alerts whenever the
recommended action is `alert` or `block`, and the action agent can also call the
`send_alert` tool with its own wording. Without a token the alert is written to the
log instead, and `dry_run: true` in the request suppresses it entirely.

### Endpoints

| method | path | purpose |
|---|---|---|
| POST | `/analyze` | Analyze a batch of events |
| GET | `/incidents` | List incidents, filterable by IP or analysis |
| GET | `/incidents/{id}` | Full verdict for one incident |
| GET | `/analyses/{id}` | Status of one analysis |
| GET | `/blocklist` | Contained IPs |
| GET | `/health` | Health check, no authentication |
| GET | `/docs` | Interactive OpenAPI documentation |

Every endpoint except `/health`, `/` and `/docs` requires the `X-API-Key` header.
Status codes: `200` when processing completed, `202` when the synchronous deadline
expired and the remaining incidents continue in the background, `401` without a
valid key, `413` above `MAX_BATCH_SIZE`, `422` for a malformed event, `503` when the
model is not loaded.

### See it work end to end

```bash
python scripts/demo.py --scenario mixed
```

Sends real events from the dataset to a running instance and prints the resulting
incidents. Scenarios: `benign`, `mixed`, `sqli`, `rce`, `scanning`.

### Configuration

Every value has a safe default; copy `.env.example` to `.env` to change any of
them. The two that matter most:

| variable | default | effect |
|---|---|---|
| `NOVELTY_TARGET_FPR` | `0.002` | Detector operating point. Lower means fewer false positives and more missed windows. |
| `BLOCK_MODE` | `dry_run` | `enforce` runs the real firewall command from `BLOCK_COMMAND`. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | empty | Where incident alerts are delivered. Empty means log only. |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | empty | Both empty means the agents fall back to the deterministic verdict. |
| `LLM_PROVIDER` | `auto` | `anthropic` or `gemini` to force one when both keys exist. |
| `LLM_MODEL` | provider default | `claude-opus-5` or `gemini-3.5-flash` unless set. |

Never commit a real token. `.env` is gitignored; `.env.example` holds placeholders
only.

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

184 tests, 93% coverage, no network access.

---

## Repository layout

| path | contents |
|---|---|
| `app/main.py` | FastAPI application: routes, authentication, batch size limit, `202` fallback |
| `app/orchestrator.py` | Runs the funnel end to end and holds the deterministic fallback verdict |
| `app/schemas.py` | Pydantic request and response contracts |
| `app/config.py` | Settings read from environment or `.env` |
| `app/db.py` | SQLite persistence for analyses, incidents, actions and blocklist |
| `app/pipeline/parser.py` | Normalizes structured events and raw Combined Log Format lines |
| `app/pipeline/novelty.py` | The character n-gram model: fit, scoring, calibration table, span localization |
| `app/pipeline/detector.py` | Loads the trained artifact, applies the operating point, derives severity |
| `app/pipeline/correlator.py` | Groups suspicious events into (IP, 5-minute window) incident dossiers |
| `app/agents/base.py` | Manual tool-use loop, provider-agnostic, with call budget |
| `app/agents/providers.py` | Gemini adapter over the same client surface the loop expects |
| `app/agents/triage.py` | Triage agent: read-only tools, terminal tool `submit_verdict` |
| `app/agents/action.py` | Action agent: actuation tools, terminal tool `submit_action_plan` |
| `app/actuators/blocker.py` | IP blocking, dry-run by default, with literal-IP validation |
| `app/actuators/notifier.py` | Telegram alerts, falling back to the log |
| `scripts/sample_dataset.py` | Turns the 2.8 GB dataset into a Git-versionable sample |
| `scripts/train_model.py` | Trains, calibrates, evaluates and writes `reports/metrics.json` |
| `scripts/baselines.py` | Regex and supervised baselines — comparison only, outside the service |
| `scripts/inspect_errors.py` | Audits false positives and negatives against independent signatures |
| `scripts/experiment_novelty.py` | The experiment that motivated the unsupervised approach |
| `scripts/demo.py` | End-to-end demonstration against a running API |
| `tests/` | 12 test files: pipeline, agents, providers, API, actuators, persistence, recall policy |
| `data/` | `sample.csv.gz` (12 MB), 303,344 events |
| `models/` | `detector.joblib` (1.7 MB), the trained artifact |
| `reports/` | `metrics.json` with every measurement quoted here |

---

## Architecture and agent flow

Four stages. The first two are deterministic and run in milliseconds; the last two
are the LLM agents.

**1. Ingestion and scoring.** Normalizes each event — structured JSON or a raw
Combined Log Format line — and scores it against the character model. The score is
normalized so 0.50 is the escalation floor.

**2. Correlation.** Groups the flagged events by (IP, 5-minute window) into
*incident dossiers*. A dossier carries aggregate statistics plus the five least
likely events with the exact substring the model objected to. Three hundred requests
become under 4,000 characters, and a window with nothing above the floor produces no
dossier at all — that is what keeps the LLM cost viable.

**3. Triage agent (LLM).** Receives one dossier and classifies it. It does not get
the raw event list: if it wants more data it calls a tool — pull raw events, regex
search the window, aggregate by field, look up the IP's history. It finishes by
calling `submit_verdict`, whose JSON schema is the output contract.

**4. Action agent (LLM).** Receives the verdict as an input and decides the
proportional response, then executes it: `block`, `rate_limit`, `alert`, `monitor` or
`allow`. It finishes by calling `submit_action_plan`.

The agents communicate through an internal flow: the correlator hands a dossier to
triage, and triage's verdict is the action agent's input. Whoever classifies does not
decide, and whoever decides does not reclassify.

Two behaviors worth knowing. A benign triage verdict ends the incident there, so it
costs one LLM call instead of two. And if the LLM is unavailable or the per-request
call budget runs out, both stages fall back to a deterministic verdict — which never
closes a case as benign, and says so in the response through
`incidents_awaiting_agent`.

Detection is unsupervised by design. The dataset's labels were generated by regex, so
training on them would only reproduce the regex; the optional fine-tuning step was
dropped for that measured reason, not by omission. The evidence is in
[DETAILS.md](DETAILS.md).

---

## Machine learning results

Held-out set: 93,023 events from 25,477 IPs, 1,195 attacks. Split by IP, never at
random. Threshold calibrated on a benign slice separate from the fit.

### Detector at the default operating point (0.2% FPR)

| metric | value |
|---|---:|
| Event recall | 0.9883 |
| Incident recall | 0.9972 |
| Precision | 0.8620 |
| PR-AUC | 0.9698 |
| ROC-AUC | 0.9994 |
| Effective false positive rate | 0.2058% |
| False positives | 189 |
| False negatives (event level) | 14 |
| Windows containing an attack that were not escalated | 1 of 362 |

### Recall by attack family

Every family is zero-shot: none of them entered the fit.

| family | n | recall | PR-AUC |
|---|---:|---:|---:|
| bot | 304 | 1.000 | 0.879 |
| rce | 66 | 1.000 | 0.906 |
| scanning | 155 | 0.987 | 0.620 |
| sqli | 670 | 0.982 | 0.971 |

### Comparison with baselines

| detector | recall | precision | uses labels? | detects unseen families? |
|---|---:|---:|---|---|
| 15-line regex | 0.844 | 0.997 | it *is* the label | no |
| Supervised classifier | 0.911 | 0.835 | yes | no |
| Novelty detection | **0.988** | 0.862 | **no** | **yes** |

The novelty detector dominates the supervised model on both metrics without using
labels. Against the regex it trades 13.5 points of precision for 14.4 points of
recall, and adds generalization to unseen attacks that a regex cannot have by
definition.

### Operating points

The calibration table ships inside the artifact, so moving the operating point takes
no retraining — set `NOVELTY_TARGET_FPR`. Every threshold comes from the held-out
benign slice, the same one the service uses.

| target FPR | event recall | incident recall | windows missed | precision | dossiers |
|---:|---:|---:|---:|---:|---:|
| 0.02% | 0.525 | 0.202 | 289 | 0.974 | 89 |
| 0.05% | 0.563 | 0.202 | 289 | 0.956 | 103 |
| 0.10% | 0.931 | 0.978 | 8 | 0.938 | 423 |
| **0.20%** | **0.988** | **0.997** | **1** | **0.862** | **532** |
| 0.50% | 0.999 | 0.997 | 1 | 0.739 | 742 |
| 1.00% | 0.999 | 0.997 | 1 | 0.579 | 1,124 |

Incident recall is the metric that matters, because a dossier carries the IP's whole
window: an event below the floor is still read by the agent when a sibling in the
same window rose above it. The curve's knee is at 0.2% — the last point where
spending more triage buys incident coverage.

---

## Further reading

- [DETAILS.md](DETAILS.md) — the data, the model, the methodology, the error
  analysis and the measured limits.
