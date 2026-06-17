# Shape — Bedesten Count-Guided Retrieval

## What this builds
Add an experimental Bedesten count-guided retrieval spike and measurement harness, not a production retrieval primitive yet. The spike tests whether Bedesten `total` counts can increase the probability of recalling buried but on-point decisions without embeddings. Production MCP exposure is gated on real-corpus target `documentId` recall, not on synthetic fixtures.

## Key interfaces

### Existing Bedesten search request
`BedestenSearchData(pageSize, pageNumber, itemTypeList, phrase, birimAdi, kararTarihiStart, kararTarihiEnd, sortFields, sortDirection)` is the request payload for Bedesten search. It supports page size, page number, court type filters, phrase syntax, chamber filter, date filters, and default date-desc sorting.

### Existing Bedesten search response
`BedestenSearchResponse.data` contains `emsalKararList`, `total`, and `start`. The new retrieval flow must treat `total` as corpus-level result count only, not as proof that any specific target decision is present.

### Existing Bedesten document fetch
`get_document_as_markdown(documentId)` retrieves full text and counts against the same Bedesten upstream rate limit as search. The new tool must keep full-text fetches small and separate from cheap search/count probes.

### New MCP tool
```python
async def search_bedesten_count_guided(
    base_query: str,
    discriminator_candidates: list[str],
    court_types: list[BedestenCourtTypeEnum],
    policy: str = "loose_pages",
    min_total_floor: int = 1,
    page_size: int = 100,
    max_probe_searches: int = 7,
    max_pages_per_final_query: int = 2,
    max_window_searches: int = 0,
    max_fulltext_fetches: int = 10,
    kararTarihiStart: str = "",
    kararTarihiEnd: str = "",
    eval_reference_date: str = "",
    birimAdi: BirimAdiEnum = "ALL",
) -> dict:
    ...
```

This interface is the proposed production shape after the spike passes measurement. During the spike, it may remain an internal helper or dev-only tool. It accepts LLM-generated discriminator candidates and explicit spike controls, but all probe, selection, pagination, and fallback decisions happen deterministically inside MCP. The eval harness must pass `policy` and `min_total_floor` explicitly for every scored run. After the gate passes, production defaults are pinned to the winning recorded eval configuration; mutalaa callers do not get to choose experimental policy values unless a later versioned contract exposes that intentionally.

### New core helper
```python
async def run_bedesten_count_guided_retrieval(
    bedesten_client: Any,
    base_query: str,
    discriminator_candidates: list[str],
    court_types: list[str],
    policy: str = "loose_pages",
    min_total_floor: int = 1,
    page_size: int = 100,
    max_probe_searches: int = 7,
    max_pages_per_final_query: int = 2,
    max_window_searches: int = 0,
    max_fulltext_fetches: int = 10,
    karar_tarihi_start: str = "",
    karar_tarihi_end: str = "",
    eval_reference_date: str = "",
    birim_adi: str = "ALL",
) -> dict:
    ...
```

Place the deterministic selection logic outside `mcp_server_main.py` so tests can exercise it directly with fake Bedesten clients.

## Response Schema v1
All spike helpers, eval harnesses, tests, and any mutalaa consumer must use this versioned envelope. The schema is owned by `yargi-mcp`; mutalaa consumes it but must not redefine it.

```python
{
    "schema_version": "bedesten_count_guided.v1",
    "status": "success" | "partial" | "no_results" | "validation_error" | "rate_limited" | "upstream_error",
    "message": str,
    "query": {
        "base_query": str,
        "selected_query": str,
        "selected_policy": "tight_page" | "loose_pages" | "windowed_loose_pages",
        "min_total_floor": int,
        "court_types": list[str],
        "birim_adi": str,
        "karar_tarihi_start": str,
        "karar_tarihi_end": str,
        "eval_reference_date": str,
    },
    "candidate_document_ids": list[str],
    "candidates": list[{
        "document_id": str,
        "rank": int,
        "page_number": int,
        "window_id": str | None,
        "court_type": str | None,
        "chamber": str | None,
        "case_no": str | None,
        "decision_no": str | None,
        "decision_date": str | None,
        "title": str | None,
        "source_url": str | None,
        "fetched_fulltext": bool,
    }],
    "fetched_documents": list[{
        "document_id": str,
        "text_preview": str,
        "source_url": str | None,
        "mime_type": str | None,
    }],
    "diagnostics": {
        "probes": list[{
            "policy_id": str,
            "phrase": str,
            "discriminators": list[str],
            "total_records": int | None,
            "page_size": int,
            "status": str,
            "rejected_reason": str | None,
        }],
        "selected_discriminators": list[str],
        "rejected_discriminators": list[{"candidate": str, "reason": str}],
        "totals": {"base": int | None, "selected": int | None},
        "pagination": {
            "page_size": int,
            "pages_requested": int,
            "pages_fetched": int,
            "max_pages_per_final_query": int,
            "complete": bool,
        },
        "windowing": {
            "used": bool,
            "max_window_searches": int,
            "windows": list[{
                "window_id": str,
                "start": str,
                "end": str,
                "total_records": int | None,
                "pages_fetched": int,
                "complete": bool,
            }],
        },
        "budget": {
            "max_probe_searches": int,
            "searches_used": int,
            "max_fulltext_fetches": int,
            "fulltexts_used": int,
            "estimated_requests": int,
            "exhausted": bool,
        },
        "truncation": {
            "is_truncated": bool,
            "reason": "none" | "page_budget" | "window_budget" | "rate_limit" | "timeout",
            "uncovered_total_estimate": int | None,
        },
        "rate_limit": {
            "hit": bool,
            "source": "none" | "local" | "upstream",
            "retry_after_seconds": int | None,
        },
        "errors": list[str],
    },
}
```

`candidate_document_ids` is the recall-relevant candidate pool for the eval gate. `fetched_documents` is only the bounded full-text subset and must not be used for recall scoring. Schema version bumps: additive optional fields keep `bedesten_count_guided.v1`; removing fields, renaming fields, changing types, or changing enum semantics requires `bedesten_count_guided.v2`.

## Narrowing Policies
The spike must implement and report this fixed policy identifier set. Eval protocol and harnesses must use these exact IDs.

- `tight_page`: probe single discriminators, prefer high-coverage reducers above `page_size` over rare single terms that already fit, then follow one cumulative greedy stack path until `min_total_floor <= total <= page_size`; reject probes with `total < min_total_floor` or `total == 0`.
- `loose_pages`: same single-probe ordering and one-path cumulative stacking, but the stop limit is `page_size * max_pages_per_final_query`; fetch configured pages before declaring incompleteness.
- `windowed_loose_pages`: same discriminator selection as `loose_pages`, then apply date-window fallback only if selected `total` still exceeds covered pages and `max_window_searches > 0`.

`min_total_floor` is a swept eval parameter, not a hidden heuristic. The pre-registered protocol must score at least `min_total_floor=1` and `min_total_floor=max(5, ceil(page_size * 0.05))`; production defaults come from the winning recorded eval run.

Greedy selection is deterministic and bounded:
1. Probe single-discriminator candidates first, up to the global probe budget.
2. Reject probes with `total is None`, `total == 0`, or `total < min_total_floor`.
3. If any single reducer remains above the active stop limit, choose the largest-total above-band reducer first, even if another rare single discriminator already fits the band.
4. Continue on one cumulative greedy stack path by trying remaining single reducers in largest-total order. Accept a stacked discriminator only when it reduces the current stack and remains `>= min_total_floor`.
5. Stop successfully when the cumulative stack reaches the active stop band. If no above-band single reducer exists, fall back to the largest-total in-band single discriminator.
6. If no valid reducing candidate exists, stop and return the base-query pool.
7. If the stop band is never reached before `max_probe_searches`, return the recall-safer base-query pool instead of a narrowed-but-still-too-large or rare single stack. Mark `diagnostics.budget.exhausted=true` only when the probe budget was actually consumed; otherwise record `no_reducing_candidate` in `diagnostics.errors`. Set `diagnostics.truncation.is_truncated=true` only if the returned base-query pool also exceeds the configured page/window coverage.

## Phrase Assembly
Plain base-query terms and atomic discriminator terms are emitted as required terms: `+term`, so count probes measure the base-query intersection plus the candidate discriminator rather than discriminator-wide corpus frequency. Multi-word discriminator candidates are split into atomic required terms by default, preserving only non-empty whitespace-delimited tokens. If a caller supplies an already-operator-bearing base query (`+`, `-`, boolean operator, or quoted phrase), preserve it instead of blindly rewriting it. Required exact phrase mode is not a default branch; it may be added only as a separately named eval dimension after a syntax-only validation that the quoted phrase is accepted by Bedesten without 400/validation failure. The spike must not mix exact-phrase and atomic-split results under the same policy ID.

## Date Windows
If `max_window_searches > 0` and no date bounds are provided, use a deterministic default span from `2000-01-01` through `eval_reference_date` when provided, otherwise through the current date in the configured runtime timezone. Authoritative replay scoring must either pass explicit `karar_tarihi_start` and `karar_tarihi_end` or pass `eval_reference_date`; it must not depend on wall-clock current date. Split the span into `max_window_searches` contiguous equal-duration windows ordered newest-to-oldest. If explicit `karar_tarihi_start` and `karar_tarihi_end` are provided, split that closed interval by the same rule. Equal-count partitioning is out of scope because Bedesten exposes counts only after search calls. Window IDs are stable strings `w0`, `w1`, ... in newest-to-oldest order.

## Data Flow
1. Validate `base_query`, court types, budgets, and `page_size`; clamp `page_size` to 100.
2. Convert plain base-query terms to required terms, then search that canonical base query once with `pageSize=100`, `pageNumber=1`, and selected court types.
3. Assemble probe phrases with explicit required-term semantics, never by plain string concatenation. Atomic discriminator terms are added with the Bedesten `+term` required operator. Multi-word candidates are split into atomic required terms by default.
4. Probe discriminators greedily, respecting the global `max_probe_searches` budget. Default and hard cap are conservative (`7`): roughly single probes plus up to three cumulative stack probes, keeping a normal run near one Bedesten rate window instead of an unbounded burst.
5. Select a high-recall candidate pool, not the tightest possible total. Reject probes with `total < min_total_floor`, and stop adding discriminators once the policy-specific stop condition is met.
6. Run final search for the selected query with `pageSize=100`. If `total` exceeds `page_size`, fetch additional pages before opening any date-window fallback.
7. If final results remain incomplete after configured pages, return a loud incomplete/truncated diagnostic. If `max_window_searches > 0`, use date-window fallback to give older decisions dedicated coverage before marking the pool incomplete.
8. Deduplicate candidates by `documentId`.
9. Fetch at most `max_fulltext_fetches` full texts, only after candidate selection, in Bedesten search order after deduplication.
10. Return candidates, fetched previews, and detailed diagnostics. Do not claim semantic ranking or target recall unless the target `documentId` is actually present.

## Design Decisions
- Treat count-guided discriminator selection as a hypothesis to be measured because Bedesten exposes `total` for each search response, but `total` does not identify target membership.
- Use `pageSize=100` by default because Bedesten models and `search_bedesten` allow page sizes up to 100.
- Use limited pagination before date-window fallback because fetching page 2 or page 3 for the same narrow query is cheaper and simpler than multiplying query windows.
- Use date windows as fallback for incomplete pools because date-desc sorting otherwise preferentially drops older decisions, which are often the decisions this spike is trying to recover.
- Use greedy selection, not beam search, because the probe budget is global and small.
- Do not optimize for the smallest total below `page_size`. Every additional AND term can exclude on-point decisions, so the spike must compare loose and tight narrowing policies on real target recall.
- Do not use embeddings, pgvector, semantic reranking, or cross-encoder reranking in this feature.
- Treat external eval claims as unverified until target `documentId` recall is measured by an automated eval.

## Production Gate
Before exposing `search_bedesten_count_guided` as a production MCP tool, create and commit a pre-registered eval protocol. The protocol must be written before the first scoring run and must define:
- Ground truth source and provenance. Target `documentId` sets must come from the external authority eval or another documented source independent of count-guided output.
- Minimum eval size: at least 7 independent real-corpus questions and at least 10 total target document IDs. A single favorable dispute is never a pass.
- Primary baseline: current lexical `search_bedesten` behavior with the same base query, court types, date filters, and default first-page candidate collection.
- Secondary baseline: request-budget-matched lexical pagination using the same base query and the same total number of Bedesten search calls consumed by count-guided, including base search, probe searches, final pages, and window searches. Full-text fetches are excluded because recall is scored before full-text fetch.
- Primary metric: target-document recall over the candidate pool before full-text fetch, calculated as recalled target IDs divided by expected target IDs.
- Pass threshold: count-guided must not reduce recall versus the primary baseline on any eval case, must improve macro recall by at least 15 percentage points over the secondary baseline, and must recall at least two additional target documents overall. Ties are not a pass.
- Failure handling: any 429, timeout, incomplete replay, or ungoverned ground-truth source invalidates the run instead of counting as retrieval failure or success.

Recall measurement must use recorded/replayed or fully cache-warmed Bedesten responses isolated from production traffic. Live uncached multi-case runs are allowed for data capture, but not as the authoritative pass/fail scoring run because shared rate limits can corrupt recall results.

## Out of Scope
- Building a vector database or persistent semantic index.
- Embedding-based semantic search.
- Analogical transferability judging.
- Replacing `search_bedesten_semantic` or `search_bedesten_deep_semantic`.
- Shipping a production MCP tool before real-corpus recall measurement.
- Guaranteeing recall of a specific decision from `total` alone.
- Importing or depending on the external Laravel/mutalaa eval files inside this repository.
