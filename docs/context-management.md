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
| `SESSION_MAX_MESSAGES` | `200` | Maximum UI records retained per session |
| `SESSION_TTL_SECONDS` | `2592000` | Session/context retention (30 days) |

On EC2, set these in the environment used by Docker Compose. The defaults work without additional
configuration. After changing a value, recreate the app container so it receives the updated
environment:

```bash
docker compose -f docker-compose.prod.yml up -d --force-recreate app
```

Do not run `docker compose down -v` unless deleting all Redis-backed sessions is intentional.
