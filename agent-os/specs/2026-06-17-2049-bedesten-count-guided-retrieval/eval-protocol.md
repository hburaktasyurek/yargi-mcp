# Eval Protocol — Bedesten Count-Guided Retrieval

This protocol must be committed before any authoritative scoring run. It defines the production gate for `search_bedesten_count_guided`.

## Ground Truth
- Use at least 7 independent real-corpus questions.
- Use at least 10 total target `documentId` values.
- Target document IDs must come from the external authority eval or another documented source independent of count-guided output.
- A single favorable dispute is never a pass.
- Cases must include `ground_truth_provenance`.

## Scored Policies
Run every case for each policy ID:
- `tight_page`
- `loose_pages`
- `windowed_loose_pages`

For each policy, score at least these `min_total_floor` values:
- `1`
- `max(5, ceil(page_size * 0.05))`

The eval harness must pass `policy`, `min_total_floor`, and `eval_reference_date` explicitly. Do not derive policy from page/window budget values.

## Baselines
- Primary baseline: current lexical `search_bedesten` first-page behavior with the same base query, court types, date filters, and page size.
- Secondary baseline: request-budget-matched lexical pagination using the same base query.
- The secondary baseline budget counts every Bedesten search call consumed by count-guided, including base search, probe searches, final pages, and window searches. Full-text fetches are excluded because recall is scored before full-text fetch.

## Metric
Primary metric is target-document recall over `candidate_document_ids` before full-text fetch:

```text
recalled target IDs / expected target IDs
```

Do not score recall from `fetched_documents`.

## Pass Threshold
Production exposure requires all of the following:
- No per-case recall regression versus the primary baseline.
- Macro recall improvement of at least 15 percentage points versus the secondary baseline.
- At least 2 additional target documents recalled overall versus the secondary baseline.
- Ties are not a pass.

## Invalid Runs
The scoring run is invalid, not failed, if any of these occur:
- HTTP 429 or local Bedesten rate-limit interruption.
- Timeout.
- Incomplete replay fixture.
- Missing or ungoverned ground-truth provenance.
- Windowed scoring without explicit date bounds or `eval_reference_date`.

## Replay Requirement
Authoritative scoring must use recorded fixtures through `httpx.MockTransport` or an equivalent replay client. Live uncached Bedesten calls are acceptable only to capture fixtures, not to decide pass/fail.
