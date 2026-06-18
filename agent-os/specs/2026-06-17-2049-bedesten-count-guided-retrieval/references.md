# References — Bedesten Count-Guided Retrieval

## Source Files Examined

### `bedesten_mcp_module/models.py`
- **Purpose:** Defines Bedesten request and response models.
- **Relevant to this spec:** Establishes the verified API fields available to count-guided retrieval.
- **Key facts:**
  - `BedestenCourtTypeEnum` allows `YARGITAYKARARI`, `DANISTAYKARAR`, `YERELHUKUK`, `ISTINAFHUKUK`, and `KYB` (lines 10-17).
  - `BedestenSearchData.pageSize` is described as results per page with range 1-100 (line 21).
  - `BedestenSearchData.pageNumber` is 1-indexed (line 22).
  - `BedestenSearchData.phrase` supports word, exact phrase, required/exclude, and boolean operators, with no wildcards or regex (line 24).
  - `BedestenSearchData` includes chamber and date filters through `birimAdi`, `kararTarihiStart`, and `kararTarihiEnd` (lines 25-31).
  - `BedestenSearchData` leaves sorting empty by default; callers must set `sortFields` and `sortDirection` explicitly when a specific order is required.
  - `BedestenDecisionEntry` metadata contains IDs, court type, chamber, case/decision numbers, decision type, date, finalization status, and no snippet/body text (lines 45-59).
  - `BedestenSearchDataResponse` includes `emsalKararList`, `total`, and `start` (lines 61-64).
  - `BedestenDocumentMarkdown` contains `documentId`, optional `markdown_content`, `source_url`, and `mime_type` (lines 87-91).

### `mcp_server_main.py`
- **Purpose:** Registers MCP tools and maps Bedesten API responses into tool responses.
- **Relevant to this spec:** Shows current Bedesten search wrapper behavior, page size clamp, date conversion, total_records output, and document fetch behavior.
- **Key facts:**
  - The `search_bedesten` tool documents unsupported wildcards, regex, fuzzy, and proximity search and recommends exact phrases for legal terms (lines 1120-1125).
  - `search_bedesten` defaults `pageSize` to 100 and constrains it to 1-100 (lines 1130 and 1142).
  - `search_bedesten` accepts `pageNumber > 1` when the first 100 results are insufficient (line 1131).
  - Date strings are converted to ISO start/end timestamps when simple dates are supplied (lines 1144-1154).
  - `search_bedesten` constructs `BedestenSearchData` with page, phrase, court types, chamber, and date filters (lines 1156-1164).
  - The tool returns `decisions`, `total_records`, `requested_page`, `page_size`, and `searched_courts` (lines 1187-1192).
  - `get_bedesten_document_markdown` states that document retrieval counts against the same Bedesten upstream rate limit as search (lines 1242-1246).
  - Document fetch handles local and upstream rate-limit errors by returning structured 429 content (lines 1261-1298).

### `bedesten_mcp_module/client.py`
- **Purpose:** Implements the Bedesten HTTP client, rate limiting, cache, search, and document fetch.
- **Relevant to this spec:** Establishes rate-limit constraints, cache behavior, and shared search/document bucket.
- **Key facts:**
  - The client uses Bedesten base URL `https://bedesten.adalet.gov.tr`, search endpoint `/emsal-karar/searchDocuments`, and document endpoint `/emsal-karar/getDocumentContent` (lines 303-305).
  - Comments document a measured limit of 10 requests per 30 seconds and default local spacing of 3.5 seconds per token (lines 307-316).
  - Cache configuration can use Redis and has separate TTLs for search and document cache entries (lines 323-329).
  - Search requests are cache-keyed by the full request payload before rate-limited HTTP calls (lines 451-463).
  - Search calls acquire the shared bucket before posting to Bedesten (lines 463-467).
  - HTTP 429 handling penalizes the shared bucket (lines 468-472 and 419-435).
  - Document fetch also checks cache, then acquires the same bucket before posting to Bedesten (lines 491-507).
  - HTML and PDF conversion are offloaded to threads to avoid blocking the event loop (lines 535-548).

### `semantic_search/deep_bedesten.py`
- **Purpose:** Implements existing deep Bedesten semantic search and controlled query expansion.
- **Relevant to this spec:** Provides patterns to keep Bedesten calls bounded and to place complex tool logic outside `mcp_server_main.py`.
- **Key facts:**
  - Existing deep semantic search allows the same five Bedesten court types (lines 24-30).
  - It supports optional lexicon loading through `BEDESTEN_DEEP_LEXICON_PATH` and caches loaded profiles by path and mtime (lines 87-137).
  - `generate_bedesten_deep_queries` is deterministic, exposes plain queries, and is testable directly (lines 162-211).
  - `_decision_metadata` maps weak Bedesten metadata into document ID, court type, chamber, case/decision numbers, date, title, and source URL (lines 224-247).
  - `search_bedesten_deep_semantic` clamps max queries, max search results, max full-text fetches, and top_k (lines 325-328).
  - Deep semantic search fetches metadata first, then fetches document markdown only after selecting candidates (lines 342-449).

### `tests/test_deep_bedesten.py`
- **Purpose:** Unit tests for deep Bedesten semantic search.
- **Relevant to this spec:** Shows local test style, fake Bedesten clients, synthetic terms, and budget/candidate-window expectations.
- **Key facts:**
  - Tests use `unittest.IsolatedAsyncioTestCase` (line 213).
  - Fake Bedesten clients return `BedestenSearchResponse` with `BedestenSearchDataResponse(total=...)` and controlled `emsalKararList` entries (lines 51-75 and 88-107).
  - Tests use synthetic legal placeholders rather than real-world user examples (lines 55-63 and 112-115).
  - Tests assert the requested court type is passed into every Bedesten search request (lines 286-290 and 326-330).
  - A test confirms agent-supplied deep semantic limits are clamped to 8 queries and 25 full-text fetches (lines 307-325).
  - A test confirms the metadata candidate window can be larger than the full-text fetch limit, allowing a relevant item at index 20 to be found before fetch limiting (lines 333-352).

### `tests/test_bedesten_semantic_tool.py`
- **Purpose:** Unit test for the current semantic MCP tool wrapper.
- **Relevant to this spec:** Shows monkeypatching of `mcp_server_main` globals and direct invocation of FastMCP tool `.fn`.
- **Key facts:**
  - Tests set `EMBEDDING_PROVIDER=local` before importing `mcp_server_main` to expose semantic tools (lines 7-11).
  - Tests monkeypatch `bedesten_client_instance` and `get_embedder` on `mcp_server_main` (lines 95-112).
  - Tests invoke a FastMCP tool through `mcp_server_main.search_bedesten_semantic.fn(...)` (lines 102-109).
  - The semantic tool test asserts `pageSize` equals `max_candidates`, relevant document IDs are fetched, and full-text fetch count stays bounded (lines 114-118).

### `tests/testbedesten_redis_coordination.py`
- **Purpose:** Unit tests for Bedesten Redis coordination, cache, and rate limit behavior.
- **Relevant to this spec:** Shows fake Redis and HTTP transport patterns for Bedesten client testing.
- **Key facts:**
  - Tests use `unittest.IsolatedAsyncioTestCase` (line 192).
  - `sample_search_request` constructs `BedestenSearchRequest` and `BedestenSearchData` directly (lines 63-71).
  - `sample_search_response` includes `emsalKararList`, `total`, and `start` in the Bedesten response shape (lines 74-93).
  - Tests attach `httpx.MockTransport` to `BedestenApiClient` for deterministic Bedesten responses (lines 173-189).
  - Cache tests verify two identical searches can produce only one upstream Bedesten call when Redis cache is enabled (lines 192-219).

## Unverified Claims
- ⚠️ UNVERIFIED: The external Laravel/mutalaa repository reportedly contains `agent-os/specs/.../references.md` and storage JSONL evals with a 0/3 failure case, a 3-target flight example, and a 7-question authority eval. These files are not present in this repository and were not read during this spec.
- ⚠️ UNVERIFIED: A manually reported query such as `+uçuş +tazminat +zarar +iptali` allegedly recalls 3/3 target decisions in one dispute. Treat this as a hypothesis until an automated target `documentId` recall eval is wired.

## Validation Boundary
- Synthetic fake-client tests can prove request construction, budget clamping, phrase assembly, pagination, windowing, deduplication, diagnostics, and rate-limit handling.
- Synthetic fake-client tests cannot prove retrieval quality or target recall because the fixture author controls which `documentId` appears for each query and total.
- Retrieval-quality validation requires live or recorded real-corpus Bedesten scenarios with known target `documentId` sets. If the external mutalaa eval files are not imported, equivalent local JSONL cases must be created from real Bedesten document IDs before production exposure.
- Ground truth must be independent of count-guided output. The eval author may transcribe external target IDs or create a documented real-corpus set from prior research records, but must not choose targets by first running count-guided and labeling its successes.
- Live uncached Bedesten runs are acceptable for capture only. Authoritative scoring must use recorded/replayed or fully cache-warmed responses so 429 penalties and shared-bucket contention cannot masquerade as recall failure.
- Prefer recorded replay fixtures over Redis cache warming for authoritative scoring because replay fixtures do not expire mid-run. Existing tests already use `httpx.MockTransport` for deterministic Bedesten responses.
- Replay-scored windowed policies must avoid wall-clock date bounds by passing explicit date filters or a fixed `eval_reference_date`.
