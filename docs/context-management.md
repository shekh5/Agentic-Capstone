# Conversation context management

The service separates chat display data from model memory. This prevents full chain traces,
token telemetry, and tool/API logs from consuming Gemini's context window or being treated as
conversation instructions.

## Redis records

For a session ID such as `abc`, the service uses:

| Key | Purpose |
| --- | --- |
| `session:abc:messages` | UI history, including optional trace data; capped and never sent directly to Gemini |
| `session:abc:context:recent` | Clean recent messages with only `role`, `text`, and `timestamp` |
| `session:abc:context:summary` | Rolling summary of older conversation turns |
| `session:abc:context:meta` | Context format version and last compaction information |
| `session:abc:metadata` | Session title and creation metadata |
| `session:abc:documents` | IDs of extracted documents attached to the session |
| `chain_trace:{request_id}` | Full observability trace, stored separately for 24 hours |

Existing sessions are migrated lazily the next time they are used. Migration reads the old UI
records but copies only conversational text and normalized roles; embedded traces are discarded
from model memory.

## Selection and compaction

The newest 16 clean messages are retained verbatim by default. When clean history passes 20
messages, older turns are merged into the rolling summary. A summary failure leaves every source
message unchanged. Redis optimistic locking prevents a compaction from overwriting messages added
by another request while summarization was running.

For every ReAct step, the service budgets context again because tool results increase the current
prompt. It reserves tokens for model output and a safety margin, always keeps the system/tool
instructions and current request, and removes older conversation memory first. Gemini's
`count_tokens` API is used when available; a conservative local estimate is the fallback.

## Priority and adaptive compression

Context is selected in priority order while the final Gemini messages remain chronological:

| Priority | Content | Selection behavior |
| --- | --- | --- |
| Critical | System/tool prompt, retrieved document passages, and current request | Document context is removed only if required content cannot fit; the current request is never removed |
| High | Latest four clean conversation messages and compact summary | Preserved after medium items |
| Medium | Remaining recent conversation messages | Oldest removed first |
| Excluded | Full traces, API telemetry, and tool logs | Never added to conversation context |

The full candidate context determines an adaptive level:

| Level | Default trigger | Behavior |
| ---: | ---: | --- |
| `0` | Below 60% | Summary and up to 16 recent messages |
| `1` | 60% | Normalize older whitespace and keep up to 12 messages |
| `2` | 80% | Compress the summary further and keep up to 8 messages |
| `3` | 90% | Keep the compact summary and latest high-priority messages only |

If the exact token count still exceeds the budget, medium messages are removed first, followed by
the summary and then the oldest high-priority message. Retrieved document context is removed only after
conversation memory; the current request remains present even if required content alone exceeds
the planned budget. Telemetry makes that overage visible.

Tool results receive a separate deterministic limit before being copied into the next model
request. Compression retains the beginning and end of a verbose value and explicitly preserves
its final numeric value, which is the value used by step references. The original `StepResult`
and Redis trace are never mutated.

The rolling summary is sent as untrusted prior conversation content with a `user` role. Recent
assistant answers use Gemini's `model` role. Conversation content is never appended to the system
instruction.

## Environment variables

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CONTEXT_INPUT_TOKEN_BUDGET` | `24000` | Total planned input budget before reserves |
| `CONTEXT_OUTPUT_TOKEN_RESERVE` | `1000` | Space reserved for the next model response |
| `CONTEXT_TOKEN_SAFETY_MARGIN` | `1000` | Buffer for tokenizer and prompt variation |
| `CONTEXT_RECENT_MESSAGES` | `16` | Recent messages kept verbatim after compaction |
| `CONTEXT_RECENT_HIGH_WATERMARK` | `20` | Message count that triggers summarization |
| `CONTEXT_HIGH_PRIORITY_MESSAGES` | `4` | Newest messages protected after medium history |
| `CONTEXT_LIGHT_COMPRESSION_RATIO` | `0.60` | Level-1 utilization threshold |
| `CONTEXT_STRONG_COMPRESSION_RATIO` | `0.80` | Level-2 utilization threshold |
| `CONTEXT_CRITICAL_RATIO` | `0.90` | Level-3 utilization threshold |
| `CONTEXT_TOOL_OUTPUT_MAX_TOKENS` | `2000` | Per-field model-facing tool-output limit |
| `SESSION_MAX_MESSAGES` | `200` | Maximum UI records retained per session |
| `SESSION_TTL_SECONDS` | `2592000` | Session/context retention (30 days) |
| `REACT_TEMPERATURE` | `0.1` | Default ReAct randomness when the user sends no override |
| `SUMMARY_TEMPERATURE` | `0.2` | Internal rolling-summary randomness; not user-controlled |

On EC2, set these in the environment used by Docker Compose. The defaults work without additional
configuration. After changing a value, recreate the app container so it receives the updated
environment:

```bash
docker compose -f docker-compose.prod.yml up -d --force-recreate app
```

Do not run `docker compose down -v` unless deleting all Redis-backed sessions is intentional.

## User-selectable ReAct temperature

Reasoning Chain mode offers four presets in the chat input: Precise (`0.0`), Reliable (`0.1`),
Balanced (`0.4`), and Exploratory (`0.7`). The browser stores the last selection in local storage
and sends it as the optional `temperature` query parameter on `/chain/run`. FastAPI accepts values
from `0.0` through `1.0`; invalid or out-of-range input receives a `422` response. Calls that omit
the parameter use `REACT_TEMPERATURE`.

The effective value is recorded on the chain trace and each ReAct `ModelCall`, and the dashboard
shows it beside token usage. The user override never affects rolling-memory summaries, which use
the separately controlled `SUMMARY_TEMPERATURE` value.

Each model call also records `context_usage`, including budget and used tokens, utilization,
included/dropped messages, priority counts, summary inclusion, compression level, compressed
tool-field count, and retrieved document chunk count. The chain trace exposes the final call's context
usage for quick inspection.
