import json
import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np

from bedesten_mcp_module.models import (
    BedestenDecisionEntry,
    BedestenDocumentMarkdown,
    BedestenItemType,
    BedestenSearchDataResponse,
    BedestenSearchResponse,
)
from semantic_search.deep_bedesten import generate_bedesten_deep_queries
from semantic_search.deep_bedesten import load_legal_expansion_profiles
from semantic_search.deep_bedesten import search_bedesten_deep_semantic


@contextmanager
def patched_env(**values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


class FakeBedestenClient:
    def __init__(self):
        self.search_requests = []
        self.documents = {
            "relevant": (
                "Uyuşmazlık alfa işleminden doğan beta zararı, gamma yükümlülüğü "
                "ve delta tazminatı kapsamında değerlendirilmiştir. Mahkeme beta "
                "zararının giderilmesine karar vermiştir."
            ),
            "irrelevant": (
                "Uyuşmazlık ticari kira sözleşmesinde kira bedelinin uyarlanması "
                "ve temerrüt nedeniyle tahliye istemine ilişkindir."
            ),
        }

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[decision("irrelevant"), decision("relevant")],
                total=2,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content=self.documents[document_id],
            source_url=f"https://mevzuat.adalet.gov.tr/ictihat/{document_id}",
            mime_type="text/html",
        )


class ManyCandidateBedestenClient:
    def __init__(self):
        self.search_requests = []
        self.fetched_ids = []

    async def search_documents(self, search_request):
        query_index = len(self.search_requests)
        self.search_requests.append(search_request)
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[
                    decision(f"doc-{query_index}-{index}")
                    for index in range(10)
                ],
                total=100,
                start=query_index * 10,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        self.fetched_ids.append(document_id)
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content=(
                "Alfa işlemi, beta zararı ve gamma yükümlülüğü nedeniyle "
                "delta tazminatı değerlendirmesi yapılmıştır."
            ),
            source_url=f"https://mevzuat.adalet.gov.tr/ictihat/{document_id}",
            mime_type="text/html",
        )


class KeywordEmbedder:
    dimension = 2
    model = "keyword-test-embedder"
    provider = "test"

    def encode_query(self, query: str, task: str = "search result"):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_documents(self, documents, titles=None):
        embeddings = []
        for document in documents:
            text = document.lower()
            if "alfa işlemi" in text and "beta zararı" in text:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return np.array(embeddings, dtype=np.float32)


class UnnormalizedMagnitudeEmbedder:
    dimension = 2
    model = "unnormalized-test-embedder"
    provider = "test"

    def encode_query(self, query: str, task: str = "search result"):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_documents(self, documents, titles=None):
        embeddings = []
        for document in documents:
            text = document.lower()
            if "alfa işlemi" in text and "beta zararı" in text:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([100.0, 100.0])
        return np.array(embeddings, dtype=np.float32)


class BedestenDeepSemanticTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_expansion_can_use_supplied_lexicon_without_hardcoded_domain(self):
        expansion = generate_bedesten_deep_queries(
            "Omega olayında zarar doğdu.",
            max_queries=4,
            expansion_profiles=[
                {
                    "name": "omega_profile",
                    "triggers": ["omega"],
                    "terms": ["omega hukuki ilişki", "sigma sorumluluğu"],
                }
            ],
        )

        self.assertEqual(expansion["matched_profiles"], ["omega_profile"])
        self.assertIn('"omega hukuki ilişki"', expansion["queries"])

    async def test_query_expansion_can_load_profiles_from_configured_json_file(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as lexicon:
            json.dump(
                [
                    {
                        "name": "theta_profile",
                        "triggers": ["theta"],
                        "terms": ["theta uyuşmazlığı", "lambda talebi"],
                    }
                ],
                lexicon,
                ensure_ascii=False,
            )
            lexicon.flush()

            with patched_env(BEDESTEN_DEEP_LEXICON_PATH=lexicon.name):
                profiles = load_legal_expansion_profiles()

        expansion = generate_bedesten_deep_queries(
            "Theta kapsamında karar arıyorum.",
            max_queries=4,
            expansion_profiles=profiles,
        )

        self.assertEqual(expansion["matched_profiles"], ["theta_profile"])
        self.assertIn('"lambda talebi"', expansion["queries"])

    async def test_deep_semantic_search_stays_in_one_court_and_ranks_matching_chunk(self):
        client = FakeBedestenClient()

        response = await search_bedesten_deep_semantic(
            bedesten_client=client,
            embedder=KeywordEmbedder(),
            question="Alfa işlemi sonrası beta zararı için benzer kararlar.",
            court_type="YARGITAYKARARI",
            seed_terms=["alfa işlemi", "beta zararı", "gamma yükümlülüğü"],
            max_queries=6,
            max_search_results=20,
            max_fulltext_fetches=2,
            top_k=2,
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["court_type"], "YARGITAYKARARI")
        self.assertGreaterEqual(response["diagnostics"]["candidate_count"], 2)
        self.assertEqual(response["diagnostics"]["fetched_count"], 2)
        self.assertEqual(response["results"][0]["document_id"], "relevant")
        self.assertTrue(
            response["results"][0]["source_url"].startswith("https://bedesten.adalet.gov.tr/"),
            response["results"][0]["source_url"],
        )
        self.assertIn("beta zararı", response["results"][0]["best_chunks"][0]["text"])
        self.assertTrue(
            any("beta zararı" in query for query in response["generated_queries"]),
            response["generated_queries"],
        )
        self.assertTrue(
            all(
                request.data.itemTypeList == ["YARGITAYKARARI"]
                for request in client.search_requests
            )
        )

    async def test_deep_semantic_search_rejects_multiple_court_types_before_searching(self):
        client = FakeBedestenClient()

        response = await search_bedesten_deep_semantic(
            bedesten_client=client,
            embedder=KeywordEmbedder(),
            question="Alfa işlemi nedeniyle beta zararı kararları",
            court_type=["YARGITAYKARARI", "DANISTAYKARAR"],
            seed_terms=["alfa işlemi", "beta zararı"],
        )

        self.assertEqual(response["status"], "validation_error")
        self.assertEqual(client.search_requests, [])

    async def test_deep_semantic_search_clamps_agent_supplied_limits(self):
        client = ManyCandidateBedestenClient()

        response = await search_bedesten_deep_semantic(
            bedesten_client=client,
            embedder=KeywordEmbedder(),
            question="Alfa işlemi beta zararı",
            court_type="ISTINAFHUKUK",
            seed_terms=["alfa işlemi", "beta zararı"],
            max_queries=99,
            max_search_results=999,
            max_fulltext_fetches=999,
            top_k=50,
        )

        self.assertEqual(response["status"], "success")
        self.assertLessEqual(len(client.search_requests), 8)
        self.assertEqual(len(client.fetched_ids), 25)
        self.assertLessEqual(len(response["results"]), 25)
        self.assertTrue(
            all(
                request.data.itemTypeList == ["ISTINAFHUKUK"]
                for request in client.search_requests
            )
        )

    async def test_deep_semantic_search_normalizes_embedder_outputs_before_scoring(self):
        client = FakeBedestenClient()

        response = await search_bedesten_deep_semantic(
            bedesten_client=client,
            embedder=UnnormalizedMagnitudeEmbedder(),
            question="Alfa işlemi sonrası beta zararı için benzer kararlar.",
            court_type="YARGITAYKARARI",
            seed_terms=["alfa işlemi", "beta zararı"],
            max_queries=2,
            max_search_results=20,
            max_fulltext_fetches=2,
            top_k=2,
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["results"][0]["document_id"], "relevant")


if __name__ == "__main__":
    unittest.main()
