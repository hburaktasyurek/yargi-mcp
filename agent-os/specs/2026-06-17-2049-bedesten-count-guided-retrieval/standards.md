# Standards — Bedesten Count-Guided Retrieval

## Naming Conventions
- Use Bedesten model field names exactly when constructing API requests: `pageSize`, `pageNumber`, `itemTypeList`, `phrase`, `birimAdi`, `kararTarihiStart`, and `kararTarihiEnd`. Example: `search_bedesten` constructs `BedestenSearchData` with these fields in `mcp_server_main.py:1156`.
- Use snake_case for internal helper parameters and response keys where existing semantic helpers do so. Example: deep semantic diagnostics use `candidate_count`, `deduped_candidate_count`, and `searched_query_count` in `semantic_search/deep_bedesten.py:296`.
- Use explicit status strings in structured responses. Existing deep semantic responses use values such as `validation_error`, `no_results`, `chunking_error`, `embedding_error`, and `success` through `_semantic_response` in `semantic_search/deep_bedesten.py:259`.
- Include `schema_version` in count-guided responses. Mutalaa-facing diagnostics must be treated as a versioned contract, not an incidental dict.
- Use `bedesten_count_guided.v1` for the first response contract. Keep additive optional fields on v1; bump to v2 for removed fields, renamed fields, type changes, or enum semantic changes.

## Structural Patterns
- Keep complex retrieval logic outside `mcp_server_main.py`. Existing deep semantic search is implemented in `semantic_search/deep_bedesten.py` and only registered from `mcp_server_main.py`.
- Inject the Bedesten client into the core helper so tests can use fake clients. Existing `search_bedesten_deep_semantic` accepts `bedesten_client` and `embedder` parameters in `semantic_search/deep_bedesten.py:276`.
- Build Bedesten requests through `BedestenSearchRequest(data=BedestenSearchData(...))`. Existing code follows this pattern in `mcp_server_main.py:1156` and `semantic_search/deep_bedesten.py:348`.
- Clamp agent-supplied limits before use. Existing `search_bedesten_deep_semantic` clamps query, search, fetch, and top-k limits in `semantic_search/deep_bedesten.py:325`.
- Return diagnostics with every success or controlled failure. Existing `_semantic_response` always includes `status`, `message`, and `diagnostics` in `semantic_search/deep_bedesten.py:259`.
- Respect Bedesten rate-limit errors and preserve structured 429 information. Existing Bedesten tools catch `BedestenRateLimited` and `httpx.HTTPStatusError` 429 in `mcp_server_main.py:1194` and `mcp_server_main.py:1215`.
- Estimate and report per-call request count before execution where possible. Search and document fetch share the same Bedesten rate bucket.
- Use explicit required-term phrase assembly for discriminators. Do not rely on plain whitespace concatenation to narrow results.
- Pre-register eval protocols before scoring. The protocol must pin baselines, minimum case count, target count, pass thresholds, ground-truth provenance, and invalid-run rules.
- Register any pre-gate MCP wrapper only when `BEDESTEN_COUNT_GUIDED_EXPERIMENTAL=1` is set. Production registration is a separate post-gate change.
- Use recorded fixtures through `httpx.MockTransport` or an equivalent replay client for authoritative eval scoring. Redis cache warming is acceptable only for exploratory capture because TTL expiry can confound scoring.
- Pass `policy`, `min_total_floor`, and `eval_reference_date` explicitly from tests and eval harnesses. Do not derive policy implicitly from page/window budget values.
- Count probe searches in request-budget-matched baseline accounting. Count-guided's probes are part of its retrieval cost.

## Anti-Patterns
- Do not use embeddings or semantic similarity in this feature. Existing semantic tools are optional and depend on embedding configuration; this retrieval tool must work without semantic availability.
- Do not treat `total` as target recall proof. `total` is a response count from `BedestenSearchDataResponse`, not a statement about any specific `documentId`.
- Do not optimize for the tightest total below the page budget by default. Additional AND terms can exclude relevant decisions, and the correct looseness must be measured on real-corpus recall.
- Do not run query x date-window x court-type as an always-on grid. Windowing is fallback-only because it multiplies search calls and consumes the same Bedesten rate budget.
- Do not fetch full text during probe selection. Full-text retrieval uses `getDocumentContent` and shares the Bedesten rate limit with search.
- Do not return quiet success when final results are truncated by page or window budget. Mark the pool incomplete and expose the uncovered total.
- Do not add real user-provided examples to tests. Existing deep semantic tests use synthetic examples such as `alfa işlemi` and `beta zararı`.
- Do not score production readiness on live uncached eval runs. A 429 or shared-bucket stall invalidates the run because it measures rate limiting, not retrieval quality.
- Do not let the same run discover targets and validate recall. Ground truth must be fixed before count-guided scoring.
- Do not let producer code, tests, eval harnesses, or mutalaa consumers invent response key names outside the v1 schema.
- Do not use wall-clock current date in authoritative replay scoring. Provide explicit date bounds or `eval_reference_date`.

## Test Conventions
- Use `unittest`, especially `unittest.IsolatedAsyncioTestCase`, for async tool/core tests. Existing examples are `tests/test_deep_bedesten.py:213` and `tests/testbedesten_redis_coordination.py:192`.
- Use fake Bedesten clients for core tests. Existing tests define fake clients returning controlled `BedestenSearchResponse` objects in `tests/test_deep_bedesten.py:51`.
- For MCP wrapper tests, monkeypatch `mcp_server_main.bedesten_client_instance` and call the FastMCP tool via `.fn`, following `tests/test_bedesten_semantic_tool.py:95`.
- Assert request details, not only returned results. Existing tests assert `request.data.itemTypeList` and `request.data.pageSize` in `tests/test_deep_bedesten.py:286` and `tests/test_deep_bedesten.py:349`.
- Add target recall evals by known real `documentId` sets. Unit tests with fake clients may assert that a planted candidate pool contains expected IDs, but those tests must be labeled as code-path validation, not retrieval-quality validation.
- Assert that `candidate_document_ids` contains the recall-scored pool and that `fetched_documents` is ignored for recall scoring.
- Test each policy ID and each required `min_total_floor` sweep value through explicit helper arguments.
