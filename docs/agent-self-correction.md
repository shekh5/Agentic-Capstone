# Bounded agent self-correction

Self-correction in this project means correcting a model-produced decision or tool input. It does
not mean changing Python source, Git state, configuration, or deployment. The runtime registry
contains only calculator, time, and weather tools.

## Decision contract

Gemini returns either a typed action or final decision:

```json
{"state":"action","reason":"Weather is required.","tool":"weather","tool_input":{"city":"Delhi"}}
```

```json
{"state":"final","satisfied":true,"final_summary":"Delhi is clear."}
```

Legacy responses without `state` and responses using `thought` are normalized before validation.
Action decisions are validated against the registered tool enum and a dedicated input model for
calculator, weather, or time. No invalid action reaches tool execution.

## Correction loop

Each ReAct decision permits the initial call and at most two correction calls. A malformed
decision does not consume a tool step because no tool ran. Correction feedback is XML-escaped,
marked untrusted, combined with the mandatory current request, and passed through the same context
budget and priority manager.

Recoverable categories are:

- empty response;
- invalid JSON;
- decision-schema failure;
- unknown tool;
- invalid tool input;
- unchanged repetition of a failed action;
- `satisfied=true` when every executed tool failed.

If both corrections fail, the chain returns a normal trace with `satisfied=false` and an explicit
missing valid decision. Transport failures such as an unavailable Gemini API remain infrastructure
errors and the HTTP route returns `502`.

## Duplicate and grounding controls

Failed actions are fingerprinted from the tool name and canonical input JSON. The same failed
fingerprint cannot execute again unchanged; Gemini must correct the input, choose another
registered action, or return an honest unsatisfied final response.

Final-answer grounding is intentionally conservative. A successful final answer is rejected when
all executed tools failed. Direct final answers remain possible when no tool was required, and a
mix of successful and failed results is not overruled by speculative semantic checks.

## Observability and bounds

The trace records step number, correction attempt, category, concise validation error, and whether
the next response corrected the problem. Model-call stages use names such as
`step_1_correction_1`. The dashboard renders corrections in the timeline and displays their total.

The independent bounds are:

- maximum eight executed tool actions;
- maximum two decision corrections per action slot;
- maximum one retry after a tool execution failure;
- circuit breaker for repeatedly failing tools.

These controls prevent infinite model or tool loops while keeping recoverable formatting and input
errors from immediately failing the request.
