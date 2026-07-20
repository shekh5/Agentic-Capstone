# Web Search Grounding

## Goal

The reasoning chain can answer questions whose facts may have changed after the model was trained.
This is fresh web grounding, not token-by-token response streaming.

## Runtime flow

1. The ReAct model selects `web_search` only for current, recent, changing, or explicitly
   web-sourced information.
2. The tool validates a trimmed query between 2 and 500 characters.
3. Gemini runs with the native Google Search grounding tool enabled.
4. The service accepts the result only when it has both answer text and at least one public
   `http` or `https` grounding source.
5. Up to eight unique sources are appended as Markdown links to the tool output.
6. Before completion, the orchestrator requires a satisfied final answer to preserve at least one
   exact URL returned by the successful search. A missing link gets a bounded correction request.
7. Chat renders the links safely. The trace dashboard shows queries, sources, token usage, status,
   and latency without executing provider-supplied HTML.

Web pages are untrusted model input. The search instruction tells Gemini to ignore instructions
inside pages and make only source-supported claims. The runtime tool registry still has no shell,
filesystem, Git, or code-editing capability, so web content cannot directly change the repository.

## Configuration

Use the same Gemini credential already required by the reasoning chain:

```bash
export GEMINI_API_KEY=your-key
export WEB_SEARCH_MODEL=gemini-3.1-flash-lite  # optional; falls back to GEMINI_MODEL
```

`GOOGLE_API_KEY` is also accepted. Production Compose maps the existing GitHub Actions
`GEMINI_API_KEY` secret to `GOOGLE_API_KEY`, so EC2 does not need a second search API secret.
After changing EC2 environment values, recreate the app container so Compose applies them.

## Cost and failure behavior

Google Search grounding may be billable under the selected Gemini model's pricing. A search is
therefore attempted only once by the generic executor. It is not silently retried. If it fails,
the trace records a controlled tool error and provider-call telemetry; the agent can answer that
current information is unavailable or deliberately choose a corrected query within its bounded
action limit.

No live provider call runs in CI. Tests mock Gemini and cover successful grounding, source
deduplication, missing sources, provider errors, non-retry behavior, and final citation correction.

## Operational checks

```bash
curl -X POST \
  "http://localhost:8000/chain/run?goal=What+are+today%27s+top+AI+updates%3F"
```

Confirm that:

- the trace contains a successful `web_search` result;
- `verify.final_summary` contains at least one source URL;
- `/dashboard` shows `GEMINI_SEARCH` telemetry and clickable sources;
- the answer returns a controlled unsatisfied result if grounding is unavailable.

Search grounding and its display requirements can evolve. Review the current Gemini Google Search
grounding documentation and pricing before public production use.
