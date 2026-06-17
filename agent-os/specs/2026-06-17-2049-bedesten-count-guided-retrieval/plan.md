# Plan — Bedesten Count-Guided Retrieval

## Acceptance Criteria
- [ ] A spike helper accepts a base query and discriminator candidates, probes Bedesten `total`, and records whether count-guided narrowing improves real-corpus target `documentId` recall compared with existing query strategies.
- [ ] The spike has an explicit measurement phase before production MCP exposure; synthetic tests validate code paths only and are not accepted as retrieval-quality evidence.
- [ ] Budget defaults other than Bedesten `pageSize=100` are treated as experimental until derived from measurement on real or equivalent live-corpus scenarios.
- [ ] The final selected query uses `pageSize=100` unless a lower value is explicitly configured, and never exceeds Bedesten's supported maximum of 100.
- [ ] Phrase assembly is explicit: plain base-query terms and discriminators are emitted as atomic required `+term` tokens by default, not plain concatenation or exact-phrase matching.
- [ ] The selection loop uses a small global probe budget and one cumulative greedy stack path; it does not run beam search, unconstrained combinations, or caller-requested probe bursts above the server cap.
- [ ] The loop rejects `total == 0` probes and avoids selecting the tightest total blindly; it optimizes for measured recall, not merely `total <= page_size`.
- [ ] If the selected query has `total > page_size`, the tool fetches additional pages before using date-window fallback.
- [ ] If configured pages and windows do not cover the selected total, the response marks the candidate pool as incomplete/truncated loudly.
- [ ] Date-window fallback runs only when limited pagination is insufficient and `max_window_searches > 0`; if disabled, truncation must be explicit.
- [ ] Full-text fetches are capped by `max_fulltext_fetches` and happen after metadata candidate selection.
- [ ] Response diagnostics follow `bedesten_count_guided.v1`; `candidate_document_ids` is the only recall-scored candidate pool field, while `fetched_documents` is only the bounded full-text subset.
- [ ] Narrowing policies are exactly `tight_page`, `loose_pages`, and `windowed_loose_pages`; task implementations, eval protocol, and reports use these identifiers without renaming.
- [ ] `policy`, `min_total_floor`, and `eval_reference_date` are explicit helper inputs so tests and eval can drive every policy/floor/window replay combination deterministically.
- [ ] Greedy selection avoids accepting rare single-discriminator fits when higher-coverage reducers can be stacked, and returns the base-query pool with truncation diagnostics if no stack reaches the stop band within budget.
- [ ] Real-corpus or equivalent live-corpus eval verifies target recall by `documentId`; tests do not assert exact query-string equality as the success condition.
- [ ] The eval protocol is pre-registered before scoring and pins ground-truth provenance, baselines, case count, target count, recall metric, pass threshold, and invalid-run conditions.
- [ ] Production exposure requires at least 7 independent real-corpus questions, at least 10 total target document IDs, no per-case recall regression versus current first-page `search_bedesten`, macro recall improvement of at least 15 percentage points versus request-budget-matched lexical pagination, and at least two additional target documents recalled overall.
- [ ] Authoritative scoring uses recorded/replayed or fully cache-warmed Bedesten responses isolated from production traffic; live uncached runs may capture data but do not decide pass/fail.
- [ ] The feature works without semantic embedding configuration.

## Implementation Tasks
1. Define schema and policy constants — `bedesten_mcp_module/count_guided.py` — add `SCHEMA_VERSION = "bedesten_count_guided.v1"` and policy IDs `tight_page`, `loose_pages`, `windowed_loose_pages` before implementing producer or consumer code.
2. Add spike core module — `bedesten_mcp_module/count_guided.py` — implement query assembly, probe result structures, greedy policy variants, pagination selection, candidate deduplication, and response diagnostics using the v1 schema.
3. Add a dev-only wrapper gated by `BEDESTEN_COUNT_GUIDED_EXPERIMENTAL=1` — `mcp_server_main.py` — and do not register the production MCP tool unless the measurement gate passes.
4. Reuse existing Bedesten models — `bedesten_mcp_module/models.py` — construct `BedestenSearchRequest` and `BedestenSearchData` with `pageSize`, `pageNumber`, `itemTypeList`, `phrase`, `birimAdi`, and date filters.
5. Implement page-size and budget clamps — `bedesten_mcp_module/count_guided.py` — clamp `page_size` to 1-100 and expose request-count estimates for probe, page, window, and full-text budgets.
6. Implement phrase assembly — `bedesten_mcp_module/count_guided.py` — convert plain base-query terms and multi-word discriminators into atomic `+term` required tokens by default; preserve already-operator-bearing base queries; do not use exact phrase mode unless the eval protocol adds a separate named dimension.
7. Implement greedy discriminator policies — `bedesten_mcp_module/count_guided.py` — implement `tight_page`, `loose_pages`, and `windowed_loose_pages` exactly as defined in shape.md, including the swept `min_total_floor`, tie-break order, and base-query fallback when no stack reaches the policy band.
8. Implement limited pagination — `bedesten_mcp_module/count_guided.py` — fetch page 2 through `max_pages_per_final_query` only for the final selected query and only when `total` exceeds the first page.
9. Implement fallback windowing — `bedesten_mcp_module/count_guided.py` — split explicit date bounds or the default `2000-01-01` to `eval_reference_date` or current-date span into equal-duration newest-to-oldest windows; enforce `max_window_searches`; emit stable window IDs `w0`, `w1`, ...
10. Implement bounded full-text fetch — `bedesten_mcp_module/count_guided.py` — fetch full text for the top deduplicated candidates in search order up to `max_fulltext_fetches`, preserve unfetched candidates in diagnostics, and return structured rate-limit errors if Bedesten raises 429.
11. Add focused unit tests — `tests/test_bedesten_count_guided.py` — use fake Bedesten clients that expose controlled totals, pages, document IDs, phrase assembly, truncation, and rate-limit conditions. These tests prove implementation behavior, not retrieval quality.
12. Add pre-registered eval protocol — `agent-os/specs/.../eval-protocol.md` or equivalent — define ground-truth provenance, minimum eval size, primary and secondary baselines, recall metric, pass threshold, invalid-run conditions, recording/replay rules, `policy` values, `min_total_floor` sweep values, `eval_reference_date`, and matched-budget accounting before any scoring run.
13. Add real-corpus eval harness — `scripts/evaluate_bedesten_count_guided.py` or equivalent — read a local JSONL of base query, discriminator candidates, court types, and target `documentId` sets; pass `policy`, `min_total_floor`, and `eval_reference_date` explicitly; use recorded fixtures through `httpx.MockTransport` or an equivalent replay client for authoritative scoring; report recall, request count, truncation, selected policy, and invalid-run reasons.
14. Update docs — `README.md` and `.env.example` if needed — document the spike as lexical/count-guided retrieval, not semantic retrieval, the `BEDESTEN_COUNT_GUIDED_EXPERIMENTAL=1` gate, and the production eval gate.

## Dependencies
- The core helper must be implemented before the MCP wrapper.
- The fake Bedesten client tests must be written before relying on live Bedesten behavior, but they do not validate the recall hypothesis.
- The eval harness must use target `documentId` recall. If external Laravel/mutalaa eval files are unavailable, the spike must reproduce equivalent real-corpus scenarios with real Bedesten document IDs from a documented source independent of count-guided output.
- The authoritative eval run must use recorded/replayed or fully cache-warmed responses. A live uncached run that hits 429, timeout, or incomplete collection is invalid, not a recall result.
- Windowed authoritative eval runs must pass explicit date bounds or `eval_reference_date` to avoid wall-clock-dependent replay misses.
- Date-window fallback depends on existing `kararTarihiStart` and `kararTarihiEnd` request fields.

## Known Risks
- `total` is only a corpus-level proxy; it does not prove any specific target decision is present.
- Each added discriminator can exclude an on-point decision that does not contain that term, so tightest-fit selection can be worse than looser coverage.
- Date-desc sorting means incomplete result pools preferentially drop older decisions.
- Weak Bedesten metadata prevents semantic reranking before full-text fetch.
- Bedesten search and document fetch share the same local/upstream rate-limit bucket.
- External review references to a 7-question eval and a 3-target flight example live outside this repository and are unverified here.
- One spike call can consume many shared Bedesten rate-limit tokens; diagnostics must estimate worst-case request count.
- Live uncached eval can confuse rate-limit failures with retrieval failures, so scoring must be replayed or cache-warmed and isolated.
- Very broad disputes may still exceed affordable pages and date windows; those must return incomplete/truncated status instead of quiet success.
