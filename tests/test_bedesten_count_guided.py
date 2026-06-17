import unittest
import asyncio
from pathlib import Path
import importlib.util

from bedesten_mcp_module.models import (
    BedestenDecisionEntry,
    BedestenDocumentMarkdown,
    BedestenItemType,
    BedestenSearchDataResponse,
    BedestenSearchResponse,
)
from bedesten_mcp_module.count_guided import (
    BedestenRateLimited,
    POLICY_LOOSE_PAGES,
    POLICY_TIGHT_PAGE,
    POLICY_WINDOWED_LOOSE_PAGES,
    SCHEMA_VERSION,
    run_bedesten_count_guided_retrieval,
)


def load_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bedesten_count_guided.py"
    spec = importlib.util.spec_from_file_location("evaluate_bedesten_count_guided", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(document_id: str, court_type: str = "YARGITAYKARARI") -> BedestenDecisionEntry:
    return BedestenDecisionEntry(
        documentId=document_id,
        itemType=BedestenItemType(name=court_type, description=court_type),
        birimAdi="11. Hukuk Dairesi",
        kararTarihi="2024-01-01T00:00:00.000Z",
        kararTarihiStr="01.01.2024",
        esasNo="2024/1",
        kararNo="2024/2",
    )


class CountGuidedFakeClient:
    def __init__(self):
        self.search_requests = []
        self.fetched_ids = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        page_number = search_request.data.pageNumber

        totals_by_phrase = {
            "+alfa +konu": 300,
            "+alfa +konu +beta +zarar": 120,
        }
        total = totals_by_phrase[phrase]
        start = (page_number - 1) * search_request.data.pageSize
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[
                    decision(f"doc-{page_number}-{index}")
                    for index in range(2)
                ],
                total=total,
                start=start,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        self.fetched_ids.append(document_id)
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content=f"{document_id} karar metni",
            source_url=f"https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}",
            mime_type="text/html",
        )


class PolicySelectionFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        total = {
            "+lambda +konu": 500,
            "+lambda +konu +dar": 10,
            "+lambda +konu +orta": 80,
            "+lambda +konu +genis": 95,
        }[phrase]
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision(f"{phrase}-doc")],
                total=total,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content="lambda karar metni",
            source_url=f"https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}",
            mime_type="text/html",
        )


class BudgetExhaustedFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        page_number = search_request.data.pageNumber
        total = {
            "+theta +konu": 500,
            "+theta +konu +bir": 300,
        }[phrase]
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision(f"base-{page_number}")],
                total=total,
                start=(page_number - 1) * search_request.data.pageSize,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content="theta karar metni",
            source_url=f"https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}",
            mime_type="text/html",
        )


class WindowedFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        start = search_request.data.kararTarihiStart
        total = 360
        if phrase == "+omega +konu +kalem":
            total = 250
        if start:
            total = 40
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[
                    decision(f"window-{len(self.search_requests)}")
                ],
                total=total,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content="omega karar metni",
            source_url=f"https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}",
            mime_type="text/html",
        )


class RateLimitedFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        raise BedestenRateLimited(retry_after=12.2)

    async def get_document_as_markdown(self, document_id: str):
        raise AssertionError("documents should not be fetched after rate limit")


class SearchFailureFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        raise RuntimeError("upstream unavailable")

    async def get_document_as_markdown(self, document_id: str):
        raise AssertionError("documents should not be fetched after upstream search error")


class FetchFailureFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision("kept-candidate")],
                total=1,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        raise RuntimeError("document fetch failed")


class NoReducingCandidateFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        total = {
            "+upsilon +konu": 100,
            "+upsilon +konu +artiran": 140,
        }[phrase]
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision("upsilon-doc")],
                total=total,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content="upsilon karar metni",
            source_url=f"https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}",
            mime_type="text/html",
        )


class RequiredBaseIntersectionFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        total = {
            "+alfa +konu": 500,
            "+alfa +konu +zarar": 80,
        }[phrase]
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision(f"{phrase}-doc")],
                total=total,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        raise AssertionError("documents should not be fetched")


class ManyProbeFakeClient:
    def __init__(self):
        self.search_requests = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        phrase = search_request.data.phrase
        total = 1000 if phrase == "+kapsamli +uyusmazlik" else 900
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision("broad-doc")],
                total=total,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        raise AssertionError("documents should not be fetched")


class BedestenCountGuidedTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_base_query_terms_are_required_for_count_probes(self):
        client = RequiredBaseIntersectionFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="alfa konu",
            discriminator_candidates=["zarar"],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_TIGHT_PAGE,
            min_total_floor=1,
            page_size=100,
            max_probe_searches=3,
            max_pages_per_final_query=1,
            max_fulltext_fetches=0,
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["query"]["base_query"], "alfa konu")
        self.assertEqual(response["query"]["selected_query"], "+alfa +konu +zarar")
        self.assertEqual(
            [request.data.phrase for request in client.search_requests],
            ["+alfa +konu", "+alfa +konu +zarar", "+alfa +konu +zarar"],
        )

    async def test_default_probe_budget_is_conservative_to_avoid_bedesten_bursts(self):
        client = ManyProbeFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="kapsamli uyusmazlik",
            discriminator_candidates=[f"aday{i}" for i in range(10)],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_TIGHT_PAGE,
            page_size=100,
            max_pages_per_final_query=1,
            max_fulltext_fetches=0,
        )

        self.assertLessEqual(len(response["diagnostics"]["probes"]), 4)
        self.assertEqual(response["diagnostics"]["budget"]["max_probe_searches"], 4)

    async def test_explicit_large_probe_budget_is_clamped_to_bedesten_safe_cap(self):
        client = ManyProbeFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="kapsamli uyusmazlik",
            discriminator_candidates=[f"aday{i}" for i in range(10)],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_TIGHT_PAGE,
            page_size=100,
            max_probe_searches=12,
            max_pages_per_final_query=1,
            max_fulltext_fetches=0,
        )

        self.assertLessEqual(len(response["diagnostics"]["probes"]), 4)
        self.assertEqual(response["diagnostics"]["budget"]["max_probe_searches"], 4)

    async def test_loose_policy_uses_v1_schema_and_required_term_candidate_pool(self):
        client = CountGuidedFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="alfa konu",
            discriminator_candidates=["beta zarar"],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_LOOSE_PAGES,
            min_total_floor=1,
            page_size=100,
            max_probe_searches=3,
            max_pages_per_final_query=2,
            max_window_searches=0,
            max_fulltext_fetches=1,
            eval_reference_date="2026-06-17",
        )

        self.assertEqual(response["schema_version"], SCHEMA_VERSION)
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["query"]["selected_policy"], POLICY_LOOSE_PAGES)
        self.assertEqual(response["query"]["selected_query"], "+alfa +konu +beta +zarar")
        self.assertEqual(response["query"]["min_total_floor"], 1)
        self.assertEqual(response["query"]["eval_reference_date"], "2026-06-17")
        self.assertEqual(len(response["candidate_document_ids"]), 4)
        self.assertEqual(len(response["fetched_documents"]), 1)
        self.assertNotEqual(response["candidate_document_ids"], [doc["document_id"] for doc in response["fetched_documents"]])
        self.assertEqual(client.search_requests[1].data.phrase, "+alfa +konu +beta +zarar")
        self.assertEqual(client.search_requests[-1].data.pageNumber, 2)
        self.assertEqual(client.fetched_ids, [response["candidate_document_ids"][0]])

    async def test_tight_policy_uses_largest_total_inside_floor_and_page_band(self):
        client = PolicySelectionFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="lambda konu",
            discriminator_candidates=["dar", "orta", "genis"],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_TIGHT_PAGE,
            min_total_floor=20,
            page_size=100,
            max_probe_searches=4,
            max_fulltext_fetches=0,
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["query"]["selected_policy"], POLICY_TIGHT_PAGE)
        self.assertEqual(response["query"]["selected_query"], "+lambda +konu +genis")
        self.assertEqual(response["diagnostics"]["selected_discriminators"], ["genis"])
        self.assertIn(
            {"candidate": "dar", "reason": "below_min_total_floor"},
            response["diagnostics"]["rejected_discriminators"],
        )

    async def test_budget_exhaustion_returns_base_query_pool_with_truncation(self):
        client = BudgetExhaustedFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="theta konu",
            discriminator_candidates=["bir"],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_TIGHT_PAGE,
            min_total_floor=1,
            page_size=100,
            max_probe_searches=1,
            max_pages_per_final_query=1,
            max_fulltext_fetches=0,
        )

        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["query"]["selected_query"], "+theta +konu")
        self.assertEqual(response["diagnostics"]["budget"]["exhausted"], True)
        self.assertEqual(response["diagnostics"]["truncation"]["reason"], "page_budget")
        self.assertEqual(client.search_requests[-1].data.phrase, "+theta +konu")

    async def test_windowed_policy_uses_eval_reference_date_and_stable_window_ids(self):
        client = WindowedFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="omega konu",
            discriminator_candidates=["kalem"],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_WINDOWED_LOOSE_PAGES,
            min_total_floor=1,
            page_size=100,
            max_probe_searches=1,
            max_pages_per_final_query=1,
            max_window_searches=2,
            max_fulltext_fetches=0,
            eval_reference_date="2020-12-31",
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["diagnostics"]["windowing"]["used"], True)
        self.assertEqual(
            [window["window_id"] for window in response["diagnostics"]["windowing"]["windows"]],
            ["w0", "w1"],
        )
        self.assertTrue(
            all(candidate["window_id"] in {None, "w0", "w1"} for candidate in response["candidates"])
        )
        window_requests = [
            request for request in client.search_requests
            if request.data.kararTarihiStart
        ]
        self.assertEqual(len(window_requests), 2)
        self.assertTrue(
            all("2020" in request.data.kararTarihiEnd or "2010" in request.data.kararTarihiEnd for request in window_requests),
            [request.data.kararTarihiEnd for request in window_requests],
        )

    async def test_validation_error_does_not_search(self):
        client = CountGuidedFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="ab",
            discriminator_candidates=["beta"],
            court_types=["YARGITAYKARARI"],
            policy="unknown_policy",
        )

        self.assertEqual(response["status"], "validation_error")
        self.assertEqual(client.search_requests, [])
        self.assertEqual(response["candidate_document_ids"], [])

    async def test_rate_limit_returns_structured_v1_envelope(self):
        client = RateLimitedFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="sigma konu",
            discriminator_candidates=["beta"],
            court_types=["YARGITAYKARARI"],
        )

        self.assertEqual(response["schema_version"], SCHEMA_VERSION)
        self.assertEqual(response["status"], "rate_limited")
        self.assertEqual(response["diagnostics"]["rate_limit"]["hit"], True)
        self.assertEqual(response["diagnostics"]["rate_limit"]["source"], "local")
        self.assertEqual(response["diagnostics"]["rate_limit"]["retry_after_seconds"], 13)
        self.assertEqual(response["diagnostics"]["truncation"]["reason"], "rate_limit")

    async def test_upstream_search_error_returns_v1_envelope(self):
        client = SearchFailureFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="tau konu",
            discriminator_candidates=["aday"],
            court_types=["YARGITAYKARARI"],
        )

        self.assertEqual(response["schema_version"], SCHEMA_VERSION)
        self.assertEqual(response["status"], "upstream_error")
        self.assertEqual(response["candidate_document_ids"], [])
        self.assertIn("upstream unavailable", response["diagnostics"]["errors"][0])

    async def test_fulltext_fetch_failure_keeps_candidate_pool(self):
        client = FetchFailureFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="phi konu",
            discriminator_candidates=["aday"],
            court_types=["YARGITAYKARARI"],
            max_fulltext_fetches=1,
        )

        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["candidate_document_ids"], ["kept-candidate"])
        self.assertEqual(response["fetched_documents"], [])
        self.assertIn("document fetch failed", response["diagnostics"]["errors"][0])

    async def test_no_reducing_candidate_is_not_budget_exhaustion(self):
        client = NoReducingCandidateFakeClient()

        response = await run_bedesten_count_guided_retrieval(
            bedesten_client=client,
            base_query="upsilon konu",
            discriminator_candidates=["artiran"],
            court_types=["YARGITAYKARARI"],
            policy=POLICY_TIGHT_PAGE,
            page_size=50,
            max_probe_searches=12,
            max_pages_per_final_query=1,
            max_fulltext_fetches=0,
        )

        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["query"]["selected_query"], "+upsilon +konu")
        self.assertEqual(response["diagnostics"]["budget"]["exhausted"], False)
        self.assertIn("no_reducing_candidate", response["diagnostics"]["errors"])


class BedestenCountGuidedEvalTests(unittest.TestCase):
    def test_gate_requires_thresholds_and_rejects_ties(self):
        eval_module = load_eval_module()
        cases = [
            {
                "case_id": "case-1",
                "target_document_ids": ["a", "b"],
                "primary_candidate_document_ids": ["a"],
                "secondary_candidate_document_ids": ["a"],
                "count_guided_candidate_document_ids": ["a", "b"],
            },
            {
                "case_id": "case-2",
                "target_document_ids": ["c"],
                "primary_candidate_document_ids": ["c"],
                "secondary_candidate_document_ids": [],
                "count_guided_candidate_document_ids": ["c"],
            },
        ]

        result = eval_module.evaluate_gate(cases)

        self.assertEqual(result["valid"], False)
        self.assertEqual(result["passed"], False)
        self.assertIn("minimum_case_count", result["invalid_reasons"])
        self.assertIn("minimum_target_count", result["invalid_reasons"])

    def test_gate_passes_only_with_no_primary_regression_and_macro_gain(self):
        eval_module = load_eval_module()
        cases = []
        for index in range(7):
            target = f"target-{index}"
            cases.append(
                {
                    "case_id": f"case-{index}",
                    "ground_truth_provenance": "recorded authority eval fixture",
                    "target_document_ids": [target, f"extra-{index}"],
                    "primary_candidate_document_ids": [target],
                    "secondary_candidate_document_ids": [target],
                    "count_guided_candidate_document_ids": [target, f"extra-{index}"],
                }
            )

        result = eval_module.evaluate_gate(cases)

        self.assertEqual(result["valid"], True)
        self.assertEqual(result["passed"], True)
        self.assertGreaterEqual(result["macro_recall_improvement_vs_secondary"], 0.15)
        self.assertGreaterEqual(result["additional_targets_recalled"], 2)

    def test_recorded_eval_runs_count_guided_and_baselines_from_fixtures(self):
        eval_module = load_eval_module()
        cases = [
            {
                "case_id": "recorded-1",
                "ground_truth_provenance": "recorded fixture",
                "base_query": "rho konu",
                "discriminator_candidates": ["ek"],
                "court_types": ["YARGITAYKARARI"],
                "target_document_ids": ["target"],
                "policy": POLICY_LOOSE_PAGES,
                "min_total_floor": 1,
                "page_size": 100,
                "max_probe_searches": 1,
                "max_pages_per_final_query": 2,
                "max_window_searches": 0,
                "eval_reference_date": "2026-06-17",
            }
        ]
        fixtures = [
            {"phrase": "rho konu", "page_number": 1, "total": 250, "document_ids": ["base-1"]},
            {"phrase": "+rho +konu", "page_number": 1, "total": 250, "document_ids": ["base-1"]},
            {"phrase": "+rho +konu +ek", "page_number": 1, "total": 120, "document_ids": ["target"]},
            {"phrase": "+rho +konu +ek", "page_number": 2, "total": 120, "document_ids": ["target-2"]},
            {"phrase": "rho konu", "page_number": 2, "total": 250, "document_ids": ["base-2"]},
            {"phrase": "rho konu", "page_number": 3, "total": 250, "document_ids": ["target"]},
            {"phrase": "rho konu", "page_number": 4, "total": 250, "document_ids": ["base-4"]},
        ]

        result = asyncio.run(eval_module.evaluate_recorded_cases(cases, fixtures))

        case_result = result["cases"][0]
        self.assertEqual(case_result["count_guided_candidate_document_ids"], ["target", "target-2"])
        self.assertEqual(case_result["primary_candidate_document_ids"], ["base-1"])
        self.assertIn("target", case_result["secondary_candidate_document_ids"])
        self.assertEqual(case_result["count_guided_searches_used"], 4)


if __name__ == "__main__":
    unittest.main()
