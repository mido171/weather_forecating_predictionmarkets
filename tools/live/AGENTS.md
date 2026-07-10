# Live-tool safety boundary

The root `AGENTS.md` applies. These are retained operator tools, not startup or
library code. They must never install dependencies automatically, infer
credentials, default to production, place orders, or start unbounded polling.
Every external action requires explicit execution acknowledgement, localhost or
exact target scope, one worker by default, hard request/runtime/retry budgets,
an external run ledger, and an exact stop condition. New reusable behavior
belongs in `packages/` or `apps/`; this directory should shrink over time.

Only scripts whose tracked imports resolve may remain here. Incomplete historical runners
belong under `legacy/incomplete-live-tools`, with their missing contracts documented.
