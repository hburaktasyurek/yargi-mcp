import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except Exception:  # pragma: no cover - dependency is present in normal runtime
    httpx = None

try:
    from bedesten_mcp_module.client import BedestenRateLimited
except Exception:  # pragma: no cover - lets unit tests run with minimal deps
    class BedestenRateLimited(Exception):
        def __init__(self, retry_after: float = 0.0):
            self.retry_after = retry_after
            super().__init__("Bedesten rate limited")

HTTPStatusError = httpx.HTTPStatusError if httpx is not None else type(
    "UnavailableHTTPStatusError",
    (Exception,),
    {},
)
from bedesten_mcp_module.models import BedestenSearchData, BedestenSearchRequest

SCHEMA_VERSION = "bedesten_count_guided.v1"

POLICY_TIGHT_PAGE = "tight_page"
POLICY_LOOSE_PAGES = "loose_pages"
POLICY_WINDOWED_LOOSE_PAGES = "windowed_loose_pages"
POLICIES = {POLICY_TIGHT_PAGE, POLICY_LOOSE_PAGES, POLICY_WINDOWED_LOOSE_PAGES}
DEFAULT_MAX_PROBE_SEARCHES = 7
MAX_PROBE_SEARCHES_HARD_CAP = 7
STACK_PROBE_RESERVE = 3

ALLOWED_COURT_TYPES = {
    "YARGITAYKARARI",
    "DANISTAYKARAR",
    "YERELHUKUK",
    "ISTINAFHUKUK",
    "KYB",
}

DEFAULT_WINDOW_START = "2000-01-01"
DOCUMENT_SOURCE_URL_TEMPLATE = (
    "https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}"
)


def _court_type_value(court_type: Any) -> str:
    return getattr(court_type, "value", str(court_type))


def _source_url(document_id: Optional[str]) -> Optional[str]:
    if not document_id:
        return None
    return DOCUMENT_SOURCE_URL_TEMPLATE.format(document_id=document_id)


def _normalize_simple_date(value: str, *, end: bool = False) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.endswith("Z") or "T" in value:
        return value
    suffix = "T23:59:59.999Z" if end else "T00:00:00.000Z"
    return f"{value}{suffix}"


def _parse_date(value: str) -> date:
    clean = value.split("T", 1)[0]
    return datetime.strptime(clean, "%Y-%m-%d").date()


def _format_date(value: date, *, end: bool = False) -> str:
    suffix = "T23:59:59.999Z" if end else "T00:00:00.000Z"
    return f"{value.isoformat()}{suffix}"


def _build_windows(
    *,
    max_window_searches: int,
    karar_tarihi_start: str,
    karar_tarihi_end: str,
    eval_reference_date: str,
) -> List[Dict[str, str]]:
    if max_window_searches <= 0:
        return []

    start = _parse_date(karar_tarihi_start or DEFAULT_WINDOW_START)
    end_source = karar_tarihi_end or eval_reference_date or date.today().isoformat()
    end = _parse_date(end_source)
    if end < start:
        start, end = end, start

    total_days = (end - start).days + 1
    window_days = max(1, math.ceil(total_days / max_window_searches))
    windows: List[Dict[str, str]] = []
    current_end = end
    for index in range(max_window_searches):
        current_start = max(start, current_end - timedelta(days=window_days - 1))
        windows.append(
            {
                "window_id": f"w{index}",
                "start": _format_date(current_start),
                "end": _format_date(current_end, end=True),
            }
        )
        if current_start <= start:
            break
        current_end = current_start - timedelta(days=1)
    return windows


def _required_terms(candidate: str) -> List[str]:
    return [f"+{part}" for part in str(candidate).split() if part.strip()]


def _required_base_query(base_query: str) -> str:
    parts = [part for part in str(base_query).split() if part.strip()]
    if not parts:
        return ""
    if any(
        part.startswith(("+", "-"))
        or part.upper() in {"AND", "OR", "NOT"}
        or '"' in part
        for part in parts
    ):
        return str(base_query).strip()
    return " ".join(_required_terms(base_query))


def _append_discriminators(base_query: str, discriminators: List[str]) -> str:
    terms: List[str] = []
    for discriminator in discriminators:
        terms.extend(_required_terms(discriminator))
    if not terms:
        return base_query.strip()
    return " ".join([base_query.strip(), *terms]).strip()


def _decision_to_candidate(decision: Any, *, rank: int, page_number: int, window_id: Optional[str]) -> Dict[str, Any]:
    document_id = getattr(decision, "documentId", None)
    item_type = getattr(decision, "itemType", None)
    court_type = getattr(item_type, "name", None) if item_type else None
    title_parts = [
        value
        for value in [
            getattr(decision, "birimAdi", None),
            getattr(decision, "esasNo", None),
            getattr(decision, "kararNo", None),
            getattr(decision, "kararTarihiStr", None),
        ]
        if value
    ]
    return {
        "document_id": document_id,
        "rank": rank,
        "page_number": page_number,
        "window_id": window_id,
        "court_type": court_type,
        "chamber": getattr(decision, "birimAdi", None),
        "case_no": getattr(decision, "esasNo", None),
        "decision_no": getattr(decision, "kararNo", None),
        "decision_date": getattr(decision, "kararTarihiStr", None),
        "title": " - ".join(title_parts) if title_parts else (f"Document {document_id}" if document_id else None),
        "source_url": _source_url(document_id),
        "fetched_fulltext": False,
    }


def _empty_response(
    *,
    status: str,
    message: str,
    base_query: str,
    selected_query: str,
    policy: str,
    min_total_floor: int,
    court_types: List[str],
    birim_adi: str,
    karar_tarihi_start: str,
    karar_tarihi_end: str,
    eval_reference_date: str,
    page_size: int,
    max_probe_searches: int,
    max_pages_per_final_query: int,
    max_window_searches: int,
    max_fulltext_fetches: int,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "message": message,
        "query": {
            "base_query": base_query,
            "selected_query": selected_query,
            "selected_policy": policy,
            "min_total_floor": min_total_floor,
            "court_types": court_types,
            "birim_adi": birim_adi,
            "karar_tarihi_start": karar_tarihi_start,
            "karar_tarihi_end": karar_tarihi_end,
            "eval_reference_date": eval_reference_date,
        },
        "candidate_document_ids": [],
        "candidates": [],
        "fetched_documents": [],
        "diagnostics": {
            "probes": [],
            "selected_discriminators": [],
            "rejected_discriminators": [],
            "totals": {"base": None, "selected": None},
            "pagination": {
                "page_size": page_size,
                "pages_requested": 0,
                "pages_fetched": 0,
                "max_pages_per_final_query": max_pages_per_final_query,
                "complete": False,
            },
            "windowing": {
                "used": False,
                "max_window_searches": max_window_searches,
                "windows": [],
            },
            "budget": {
                "max_probe_searches": max_probe_searches,
                "searches_used": 0,
                "max_fulltext_fetches": max_fulltext_fetches,
                "fulltexts_used": 0,
                "estimated_requests": 0,
                "exhausted": False,
            },
            "truncation": {
                "is_truncated": False,
                "reason": "none",
                "uncovered_total_estimate": None,
            },
            "rate_limit": {
                "hit": False,
                "source": "none",
                "retry_after_seconds": None,
            },
            "errors": [],
        },
    }


def _response_template(
    *,
    base_query: str,
    selected_query: str,
    policy: str,
    min_total_floor: int,
    court_types: List[str],
    birim_adi: str,
    karar_tarihi_start: str,
    karar_tarihi_end: str,
    eval_reference_date: str,
    page_size: int,
    max_probe_searches: int,
    max_pages_per_final_query: int,
    max_window_searches: int,
    max_fulltext_fetches: int,
) -> Dict[str, Any]:
    return _empty_response(
        status="success",
        message="Count-guided Bedesten retrieval completed.",
        base_query=base_query,
        selected_query=selected_query,
        policy=policy,
        min_total_floor=min_total_floor,
        court_types=court_types,
        birim_adi=birim_adi,
        karar_tarihi_start=karar_tarihi_start,
        karar_tarihi_end=karar_tarihi_end,
        eval_reference_date=eval_reference_date,
        page_size=page_size,
        max_probe_searches=max_probe_searches,
        max_pages_per_final_query=max_pages_per_final_query,
        max_window_searches=max_window_searches,
        max_fulltext_fetches=max_fulltext_fetches,
    )


async def _search(
    bedesten_client: Any,
    *,
    phrase: str,
    court_types: List[str],
    page_size: int,
    page_number: int,
    birim_adi: str,
    karar_tarihi_start: str,
    karar_tarihi_end: str,
) -> Any:
    return await bedesten_client.search_documents(
        BedestenSearchRequest(
            data=BedestenSearchData(
                pageSize=page_size,
                pageNumber=page_number,
                itemTypeList=court_types,
                phrase=phrase,
                birimAdi=birim_adi,
                kararTarihiStart=_normalize_simple_date(karar_tarihi_start),
                kararTarihiEnd=_normalize_simple_date(karar_tarihi_end, end=True),
            )
        )
    )


def _response_total(search_response: Any) -> int:
    if search_response.data is None:
        return 0
    return int(getattr(search_response.data, "total", 0) or 0)


def _response_decisions(search_response: Any) -> List[Any]:
    if search_response.data is None:
        return []
    return list(getattr(search_response.data, "emsalKararList", None) or [])


def _policy_limit(policy: str, page_size: int, max_pages_per_final_query: int) -> int:
    if policy == POLICY_TIGHT_PAGE:
        return page_size
    return page_size * max_pages_per_final_query


async def _probe_policy(
    bedesten_client: Any,
    *,
    base_query: str,
    discriminator_candidates: List[str],
    court_types: List[str],
    policy: str,
    min_total_floor: int,
    page_size: int,
    max_probe_searches: int,
    max_pages_per_final_query: int,
    birim_adi: str,
    karar_tarihi_start: str,
    karar_tarihi_end: str,
    diagnostics: Dict[str, Any],
    current_total: int,
) -> Tuple[str, List[str], Optional[int], bool]:
    selected: List[str] = []
    remaining = list(discriminator_candidates)
    stop_limit = _policy_limit(policy, page_size, max_pages_per_final_query)
    stop_reached = False
    selected_total: Optional[int] = None
    saw_reducer = False
    attempted_stack = False

    async def run_probe(
        *,
        index: int,
        candidate: str,
        stack: List[str],
    ) -> Optional[Tuple[int, str, str, int]]:
        phrase = _append_discriminators(base_query, stack + [candidate])
        search_response = await _search(
            bedesten_client,
            phrase=phrase,
            court_types=court_types,
            page_size=page_size,
            page_number=1,
            birim_adi=birim_adi,
            karar_tarihi_start=karar_tarihi_start,
            karar_tarihi_end=karar_tarihi_end,
        )
        diagnostics["budget"]["searches_used"] += 1
        total = _response_total(search_response)
        rejected_reason = None
        status = "valid"
        if total == 0:
            status = "rejected"
            rejected_reason = "zero_total"
        elif total < min_total_floor:
            status = "rejected"
            rejected_reason = "below_min_total_floor"
        discriminators = stack + [candidate]
        diagnostics["probes"].append(
            {
                "policy_id": policy,
                "phrase": phrase,
                "discriminators": discriminators,
                "total_records": total,
                "page_size": page_size,
                "status": status,
                "rejected_reason": rejected_reason,
            }
        )
        if rejected_reason:
            diagnostics["rejected_discriminators"].append(
                {"candidate": candidate, "reason": rejected_reason}
            )
            return None
        return index, candidate, phrase, total

    single_results = []
    single_probe_cap = min(len(remaining), max_probe_searches)
    if len(remaining) >= max_probe_searches:
        single_probe_cap = max(1, max_probe_searches - STACK_PROBE_RESERVE)
    for index, candidate in enumerate(remaining):
        if (
            len(diagnostics["probes"]) >= single_probe_cap
            or diagnostics["budget"]["searches_used"] >= max_probe_searches + 1
        ):
            break
        result = await run_probe(index=index, candidate=candidate, stack=[])
        if result:
            single_results.append(result)

    reducers = [
        result for result in single_results
        if min_total_floor <= result[3] < current_total
    ]
    if reducers:
        saw_reducer = True
        above_band = [
            result for result in reducers
            if result[3] > stop_limit
        ]
        if above_band:
            index, candidate, phrase, total = sorted(
                above_band,
                key=lambda item: (-item[3], item[0]),
            )[0]
            selected.append(candidate)
            selected_total = total
            current_total = total

            for next_index, next_candidate, _single_phrase, _single_total in sorted(
                [result for result in reducers if result[1] not in selected],
                key=lambda item: (-item[3], item[0]),
            ):
                if diagnostics["budget"]["searches_used"] >= max_probe_searches + 1:
                    break
                attempted_stack = True
                stack_result = await run_probe(
                    index=next_index,
                    candidate=next_candidate,
                    stack=selected,
                )
                if not stack_result:
                    continue
                _index, stack_candidate, stack_phrase, stack_total = stack_result
                if stack_total >= current_total:
                    continue
                selected.append(stack_candidate)
                selected_total = stack_total
                current_total = stack_total
                if stack_total <= stop_limit:
                    stop_reached = True
                    return stack_phrase, selected, selected_total, stop_reached
        else:
            in_band = [
                result for result in reducers
                if result[3] <= stop_limit
            ]
            if in_band:
                index, candidate, phrase, total = sorted(
                    in_band,
                    key=lambda item: (-item[3], item[0]),
                )[0]
                selected.append(candidate)
                selected_total = total
                stop_reached = True
                return phrase, selected, selected_total, stop_reached

    if diagnostics["budget"]["searches_used"] >= max_probe_searches + 1:
        diagnostics["budget"]["exhausted"] = True
        if saw_reducer or attempted_stack:
            diagnostics["errors"].append("no_stack_reached_stop_band")
    elif saw_reducer or attempted_stack:
        diagnostics["errors"].append("no_stack_reached_stop_band")
    else:
        diagnostics["errors"].append("no_reducing_candidate")
    return base_query, [], selected_total, stop_reached


def _add_candidates(
    candidates_by_id: Dict[str, Dict[str, Any]],
    decisions: List[Any],
    *,
    page_number: int,
    window_id: Optional[str],
) -> None:
    for decision in decisions:
        candidate = _decision_to_candidate(
            decision,
            rank=len(candidates_by_id) + 1,
            page_number=page_number,
            window_id=window_id,
        )
        document_id = candidate.get("document_id")
        if not document_id or document_id in candidates_by_id:
            continue
        candidates_by_id[document_id] = candidate


async def run_bedesten_count_guided_retrieval(
    *,
    bedesten_client: Any,
    base_query: str,
    discriminator_candidates: List[str],
    court_types: List[str],
    policy: str = POLICY_LOOSE_PAGES,
    min_total_floor: int = 1,
    page_size: int = 100,
    max_probe_searches: int = DEFAULT_MAX_PROBE_SEARCHES,
    max_pages_per_final_query: int = 2,
    max_window_searches: int = 0,
    max_fulltext_fetches: int = 10,
    karar_tarihi_start: str = "",
    karar_tarihi_end: str = "",
    eval_reference_date: str = "",
    birim_adi: str = "ALL",
) -> Dict[str, Any]:
    base_query = (base_query or "").strip()
    policy = str(policy or "").strip()
    min_total_floor = max(1, int(min_total_floor))
    page_size = max(1, min(int(page_size), 100))
    max_probe_searches = max(0, min(int(max_probe_searches), MAX_PROBE_SEARCHES_HARD_CAP))
    max_pages_per_final_query = max(1, min(int(max_pages_per_final_query), 10))
    max_window_searches = max(0, min(int(max_window_searches), 20))
    max_fulltext_fetches = max(0, min(int(max_fulltext_fetches), 50))
    court_type_values = [_court_type_value(court_type) for court_type in (court_types or [])]
    discriminator_candidates = [
        str(candidate).strip()
        for candidate in (discriminator_candidates or [])
        if str(candidate).strip()
    ]

    if (
        len(base_query) < 3
        or policy not in POLICIES
        or not court_type_values
        or any(court_type not in ALLOWED_COURT_TYPES for court_type in court_type_values)
    ):
        return _empty_response(
            status="validation_error",
            message="Invalid count-guided Bedesten retrieval input.",
            base_query=base_query,
            selected_query=base_query,
            policy=policy or POLICY_LOOSE_PAGES,
            min_total_floor=min_total_floor,
            court_types=court_type_values,
            birim_adi=birim_adi,
            karar_tarihi_start=karar_tarihi_start,
            karar_tarihi_end=karar_tarihi_end,
            eval_reference_date=eval_reference_date,
            page_size=page_size,
            max_probe_searches=max_probe_searches,
            max_pages_per_final_query=max_pages_per_final_query,
            max_window_searches=max_window_searches,
            max_fulltext_fetches=max_fulltext_fetches,
        )

    response = _response_template(
        base_query=base_query,
        selected_query=_required_base_query(base_query),
        policy=policy,
        min_total_floor=min_total_floor,
        court_types=court_type_values,
        birim_adi=birim_adi,
        karar_tarihi_start=karar_tarihi_start,
        karar_tarihi_end=karar_tarihi_end,
        eval_reference_date=eval_reference_date,
        page_size=page_size,
        max_probe_searches=max_probe_searches,
        max_pages_per_final_query=max_pages_per_final_query,
        max_window_searches=max_window_searches,
        max_fulltext_fetches=max_fulltext_fetches,
    )
    diagnostics = response["diagnostics"]
    required_base_query = response["query"]["selected_query"]

    try:
        base_search = await _search(
            bedesten_client,
            phrase=required_base_query,
            court_types=court_type_values,
            page_size=page_size,
            page_number=1,
            birim_adi=birim_adi,
            karar_tarihi_start=karar_tarihi_start,
            karar_tarihi_end=karar_tarihi_end,
        )
        diagnostics["budget"]["searches_used"] += 1
        base_total = _response_total(base_search)
        diagnostics["totals"]["base"] = base_total

        selected_query = required_base_query
        selected_discriminators: List[str] = []
        selected_total = base_total
        base_fits_policy = min_total_floor <= base_total <= _policy_limit(
            policy,
            page_size,
            max_pages_per_final_query,
        )
        if discriminator_candidates and max_probe_searches > 0 and not base_fits_policy:
            selected_query, selected_discriminators, probed_total, stop_reached = await _probe_policy(
                bedesten_client,
                base_query=required_base_query,
                discriminator_candidates=discriminator_candidates,
                court_types=court_type_values,
                policy=policy,
                min_total_floor=min_total_floor,
                page_size=page_size,
                max_probe_searches=max_probe_searches,
                max_pages_per_final_query=max_pages_per_final_query,
                birim_adi=birim_adi,
                karar_tarihi_start=karar_tarihi_start,
                karar_tarihi_end=karar_tarihi_end,
                diagnostics=diagnostics,
                current_total=base_total,
            )
            if stop_reached:
                selected_total = probed_total if probed_total is not None else base_total
            else:
                selected_query = required_base_query
                selected_discriminators = []
                selected_total = base_total

        response["query"]["selected_query"] = selected_query
        diagnostics["selected_discriminators"] = selected_discriminators
        diagnostics["totals"]["selected"] = selected_total

        candidates_by_id: Dict[str, Dict[str, Any]] = {}
        total_pages_available = max(1, math.ceil(selected_total / page_size)) if selected_total else 1
        pages_to_fetch = min(max_pages_per_final_query, total_pages_available)
        final_responses = []
        for page_number in range(1, pages_to_fetch + 1):
            final_response = await _search(
                bedesten_client,
                phrase=selected_query,
                court_types=court_type_values,
                page_size=page_size,
                page_number=page_number,
                birim_adi=birim_adi,
                karar_tarihi_start=karar_tarihi_start,
                karar_tarihi_end=karar_tarihi_end,
            )
            diagnostics["budget"]["searches_used"] += 1
            final_responses.append(final_response)
            _add_candidates(
                candidates_by_id,
                _response_decisions(final_response),
                page_number=page_number,
                window_id=None,
            )

        pagination_complete = selected_total <= page_size * pages_to_fetch
        diagnostics["pagination"].update(
            {
                "pages_requested": pages_to_fetch,
                "pages_fetched": len(final_responses),
                "complete": pagination_complete,
            }
        )

        if not pagination_complete and policy == POLICY_WINDOWED_LOOSE_PAGES and max_window_searches > 0:
            diagnostics["windowing"]["used"] = True
            for window in _build_windows(
                max_window_searches=max_window_searches,
                karar_tarihi_start=karar_tarihi_start,
                karar_tarihi_end=karar_tarihi_end,
                eval_reference_date=eval_reference_date,
            ):
                window_response = await _search(
                    bedesten_client,
                    phrase=selected_query,
                    court_types=court_type_values,
                    page_size=page_size,
                    page_number=1,
                    birim_adi=birim_adi,
                    karar_tarihi_start=window["start"],
                    karar_tarihi_end=window["end"],
                )
                diagnostics["budget"]["searches_used"] += 1
                window_total = _response_total(window_response)
                window_complete = window_total <= page_size
                diagnostics["windowing"]["windows"].append(
                    {
                        "window_id": window["window_id"],
                        "start": window["start"],
                        "end": window["end"],
                        "total_records": window_total,
                        "pages_fetched": 1,
                        "complete": window_complete,
                    }
                )
                _add_candidates(
                    candidates_by_id,
                    _response_decisions(window_response),
                    page_number=1,
                    window_id=window["window_id"],
                )
            pagination_complete = all(
                window["complete"] for window in diagnostics["windowing"]["windows"]
            )

        if not pagination_complete:
            response["status"] = "partial"
            response["message"] = "Candidate pool is incomplete under the configured page/window budget."
            diagnostics["truncation"] = {
                "is_truncated": True,
                "reason": "window_budget" if diagnostics["windowing"]["used"] else "page_budget",
                "uncovered_total_estimate": max(0, selected_total - len(candidates_by_id)),
            }

        response["candidates"] = list(candidates_by_id.values())
        response["candidate_document_ids"] = list(candidates_by_id.keys())
        if not candidates_by_id:
            response["status"] = "no_results"
            response["message"] = "No candidate documents found."

        fetched_documents = []
        for candidate in response["candidates"][:max_fulltext_fetches]:
            document_id = candidate["document_id"]
            try:
                document = await bedesten_client.get_document_as_markdown(document_id)
                diagnostics["budget"]["fulltexts_used"] += 1
                candidate["fetched_fulltext"] = True
                markdown = document.markdown_content or ""
                fetched_documents.append(
                    {
                        "document_id": document_id,
                        "text_preview": markdown[:700],
                        "source_url": document.source_url,
                        "mime_type": document.mime_type,
                    }
                )
            except BedestenRateLimited as e:
                diagnostics["rate_limit"] = {
                    "hit": True,
                    "source": "local",
                    "retry_after_seconds": int(math.ceil(e.retry_after)),
                }
                diagnostics["errors"].append("fulltext_fetch_rate_limited")
                response["status"] = "partial"
                response["message"] = "Candidate pool returned, but full-text fetch was rate-limited."
                break
            except HTTPStatusError as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    diagnostics["rate_limit"] = {
                        "hit": True,
                        "source": "upstream",
                        "retry_after_seconds": int(retry_after) if retry_after and retry_after.isdigit() else None,
                    }
                    diagnostics["errors"].append("fulltext_fetch_rate_limited")
                else:
                    diagnostics["errors"].append(f"fulltext_fetch_error: {e}")
                response["status"] = "partial"
                response["message"] = "Candidate pool returned, but full-text fetch failed."
                break
            except Exception as e:
                diagnostics["errors"].append(str(e))
                response["status"] = "partial"
                response["message"] = "Candidate pool returned, but full-text fetch failed."
                break
        response["fetched_documents"] = fetched_documents
        diagnostics["budget"]["estimated_requests"] = (
            diagnostics["budget"]["searches_used"] + diagnostics["budget"]["fulltexts_used"]
        )
        return response

    except BedestenRateLimited as e:
        diagnostics["rate_limit"] = {
            "hit": True,
            "source": "local",
            "retry_after_seconds": int(math.ceil(e.retry_after)),
        }
        diagnostics["truncation"] = {
            "is_truncated": True,
            "reason": "rate_limit",
            "uncovered_total_estimate": None,
        }
        diagnostics["errors"].append("rate_limit_exceeded")
        response["status"] = "rate_limited"
        response["message"] = "Bedesten local rate limit exceeded."
        return response
    except HTTPStatusError as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 429:
            retry_after = e.response.headers.get("Retry-After")
            diagnostics["rate_limit"] = {
                "hit": True,
                "source": "upstream",
                "retry_after_seconds": int(retry_after) if retry_after and retry_after.isdigit() else None,
            }
            diagnostics["truncation"] = {
                "is_truncated": True,
                "reason": "rate_limit",
                "uncovered_total_estimate": None,
            }
            diagnostics["errors"].append("rate_limit_exceeded")
            response["status"] = "rate_limited"
            response["message"] = "Bedesten upstream rate limit exceeded."
            return response
        diagnostics["errors"].append(f"upstream_http_error: {e}")
        response["status"] = "upstream_error"
        response["message"] = "Bedesten upstream HTTP error."
        return response
    except Exception as e:
        diagnostics["errors"].append(str(e))
        response["status"] = "upstream_error"
        response["message"] = "Bedesten upstream error."
        return response
