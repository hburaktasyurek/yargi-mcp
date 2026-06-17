import argparse
import asyncio
import json
from typing import Any, Dict, Iterable, List, Set

from bedesten_mcp_module.count_guided import (
    POLICY_LOOSE_PAGES,
    run_bedesten_count_guided_retrieval,
)
from bedesten_mcp_module.models import (
    BedestenDecisionEntry,
    BedestenDocumentMarkdown,
    BedestenItemType,
    BedestenSearchData,
    BedestenSearchDataResponse,
    BedestenSearchRequest,
    BedestenSearchResponse,
)

MIN_CASE_COUNT = 7
MIN_TARGET_COUNT = 10
MIN_MACRO_IMPROVEMENT = 0.15
MIN_ADDITIONAL_TARGETS = 2


def _ids(values: Iterable[Any]) -> Set[str]:
    return {str(value) for value in values if str(value)}


def recall(candidate_document_ids: Iterable[Any], target_document_ids: Iterable[Any]) -> float:
    targets = _ids(target_document_ids)
    if not targets:
        return 0.0
    candidates = _ids(candidate_document_ids)
    return len(targets & candidates) / len(targets)


def _case_metrics(case: Dict[str, Any]) -> Dict[str, Any]:
    targets = _ids(case.get("target_document_ids", []))
    primary_recall = recall(case.get("primary_candidate_document_ids", []), targets)
    secondary_recall = recall(case.get("secondary_candidate_document_ids", []), targets)
    count_guided_recall = recall(case.get("count_guided_candidate_document_ids", []), targets)
    additional_targets = len(
        (targets & _ids(case.get("count_guided_candidate_document_ids", [])))
        - (targets & _ids(case.get("secondary_candidate_document_ids", [])))
    )
    return {
        "case_id": case.get("case_id", ""),
        "target_count": len(targets),
        "primary_recall": primary_recall,
        "secondary_recall": secondary_recall,
        "count_guided_recall": count_guided_recall,
        "regressed_vs_primary": count_guided_recall < primary_recall,
        "additional_targets_vs_secondary": additional_targets,
    }


def evaluate_gate(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [_case_metrics(case) for case in cases]
    target_count = sum(metric["target_count"] for metric in metrics)
    invalid_reasons: List[str] = []
    if len(cases) < MIN_CASE_COUNT:
        invalid_reasons.append("minimum_case_count")
    if target_count < MIN_TARGET_COUNT:
        invalid_reasons.append("minimum_target_count")
    if any(case.get("invalid_run") for case in cases):
        invalid_reasons.append("invalid_run")
    if any(not case.get("ground_truth_provenance") for case in cases):
        invalid_reasons.append("missing_ground_truth_provenance")

    macro_count_guided = (
        sum(metric["count_guided_recall"] for metric in metrics) / len(metrics)
        if metrics
        else 0.0
    )
    macro_secondary = (
        sum(metric["secondary_recall"] for metric in metrics) / len(metrics)
        if metrics
        else 0.0
    )
    macro_improvement = macro_count_guided - macro_secondary
    additional_targets = sum(metric["additional_targets_vs_secondary"] for metric in metrics)
    primary_regressions = [
        metric["case_id"] for metric in metrics if metric["regressed_vs_primary"]
    ]

    passed = (
        not invalid_reasons
        and not primary_regressions
        and macro_improvement >= MIN_MACRO_IMPROVEMENT
        and additional_targets >= MIN_ADDITIONAL_TARGETS
    )
    return {
        "valid": not invalid_reasons,
        "passed": passed,
        "case_count": len(cases),
        "target_count": target_count,
        "macro_recall_count_guided": macro_count_guided,
        "macro_recall_secondary": macro_secondary,
        "macro_recall_improvement_vs_secondary": macro_improvement,
        "additional_targets_recalled": additional_targets,
        "primary_regression_case_ids": primary_regressions,
        "invalid_reasons": invalid_reasons,
        "case_metrics": metrics,
    }


def _fixture_key_from_values(
    *,
    phrase: str,
    page_number: int,
    karar_tarihi_start: str = "",
    karar_tarihi_end: str = "",
) -> tuple[str, int, str, str]:
    return (
        phrase,
        int(page_number),
        karar_tarihi_start or "",
        karar_tarihi_end or "",
    )


def _fixture_key_from_request(search_request: BedestenSearchRequest) -> tuple[str, int, str, str]:
    data = search_request.data
    return _fixture_key_from_values(
        phrase=data.phrase,
        page_number=data.pageNumber,
        karar_tarihi_start=data.kararTarihiStart or "",
        karar_tarihi_end=data.kararTarihiEnd or "",
    )


def _decision(document_id: str, court_type: str = "YARGITAYKARARI") -> BedestenDecisionEntry:
    return BedestenDecisionEntry(
        documentId=document_id,
        itemType=BedestenItemType(name=court_type, description=court_type),
        birimAdi=None,
        kararTarihi="",
        kararTarihiStr="",
    )


class RecordedBedestenClient:
    def __init__(self, fixtures: List[Dict[str, Any]]):
        self.search_requests: List[BedestenSearchRequest] = []
        self._fixtures: Dict[tuple[str, int, str, str], Dict[str, Any]] = {}
        for fixture in fixtures:
            key = _fixture_key_from_values(
                phrase=fixture["phrase"],
                page_number=fixture.get("page_number", 1),
                karar_tarihi_start=fixture.get("karar_tarihi_start", ""),
                karar_tarihi_end=fixture.get("karar_tarihi_end", ""),
            )
            self._fixtures[key] = fixture

    async def search_documents(self, search_request: BedestenSearchRequest) -> BedestenSearchResponse:
        self.search_requests.append(search_request)
        key = _fixture_key_from_request(search_request)
        if key not in self._fixtures:
            raise KeyError(f"missing recorded fixture for {key!r}")
        fixture = self._fixtures[key]
        court_type = (search_request.data.itemTypeList or ["YARGITAYKARARI"])[0]
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[
                    _decision(document_id, court_type)
                    for document_id in fixture.get("document_ids", [])
                ],
                total=int(fixture.get("total", 0)),
                start=(search_request.data.pageNumber - 1) * search_request.data.pageSize,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str) -> BedestenDocumentMarkdown:
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content="",
            source_url=f"https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}",
            mime_type="text/html",
        )


async def _baseline_candidates(
    client: RecordedBedestenClient,
    *,
    base_query: str,
    court_types: List[str],
    page_size: int,
    pages: int,
) -> List[str]:
    candidate_ids: List[str] = []
    seen = set()
    for page_number in range(1, pages + 1):
        response = await client.search_documents(
            BedestenSearchRequest(
                data=BedestenSearchData(
                    pageSize=page_size,
                    pageNumber=page_number,
                    itemTypeList=court_types,
                    phrase=base_query,
                )
            )
        )
        decisions = response.data.emsalKararList if response.data else []
        for item in decisions:
            if item.documentId not in seen:
                seen.add(item.documentId)
                candidate_ids.append(item.documentId)
    return candidate_ids


async def evaluate_recorded_cases(
    cases: List[Dict[str, Any]],
    fixtures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evaluated_cases: List[Dict[str, Any]] = []
    for case in cases:
        count_guided_client = RecordedBedestenClient(fixtures)
        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=count_guided_client,
            base_query=case["base_query"],
            discriminator_candidates=case.get("discriminator_candidates", []),
            court_types=case.get("court_types", ["YARGITAYKARARI"]),
            policy=case.get("policy", POLICY_LOOSE_PAGES),
            min_total_floor=case.get("min_total_floor", 1),
            page_size=case.get("page_size", 100),
            max_probe_searches=case.get("max_probe_searches", 12),
            max_pages_per_final_query=case.get("max_pages_per_final_query", 2),
            max_window_searches=case.get("max_window_searches", 0),
            max_fulltext_fetches=0,
            karar_tarihi_start=case.get("karar_tarihi_start", ""),
            karar_tarihi_end=case.get("karar_tarihi_end", ""),
            eval_reference_date=case.get("eval_reference_date", ""),
            birim_adi=case.get("birim_adi", "ALL"),
        )
        searches_used = response["diagnostics"]["budget"]["searches_used"]

        primary_ids = await _baseline_candidates(
            RecordedBedestenClient(fixtures),
            base_query=case["base_query"],
            court_types=case.get("court_types", ["YARGITAYKARARI"]),
            page_size=case.get("page_size", 100),
            pages=1,
        )
        secondary_ids = await _baseline_candidates(
            RecordedBedestenClient(fixtures),
            base_query=case["base_query"],
            court_types=case.get("court_types", ["YARGITAYKARARI"]),
            page_size=case.get("page_size", 100),
            pages=max(1, searches_used),
        )

        evaluated = {
            **case,
            "primary_candidate_document_ids": primary_ids,
            "secondary_candidate_document_ids": secondary_ids,
            "count_guided_candidate_document_ids": response["candidate_document_ids"],
            "count_guided_searches_used": searches_used,
            "count_guided_status": response["status"],
            "invalid_run": response["status"] == "rate_limited",
        }
        evaluated_cases.append(evaluated)

    gate = evaluate_gate(evaluated_cases)
    return {
        **gate,
        "cases": evaluated_cases,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                cases.append(json.loads(stripped))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate recorded Bedesten count-guided retrieval recall gate."
    )
    parser.add_argument("jsonl_path", help="JSONL cases with target and candidate document IDs")
    parser.add_argument("--fixtures-json", default="", help="Recorded Bedesten fixture JSON file")
    args = parser.parse_args()
    cases = load_jsonl(args.jsonl_path)
    if args.fixtures_json:
        with open(args.fixtures_json, "r", encoding="utf-8") as handle:
            fixtures = json.load(handle)
        result = asyncio.run(evaluate_recorded_cases(cases, fixtures))
    else:
        result = evaluate_gate(cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
