import importlib
import os
import unittest

import numpy as np

os.environ["EMBEDDING_PROVIDER"] = "local"

mcp_server_main = importlib.import_module("mcp_server_main")
if not hasattr(mcp_server_main, "search_bedesten_semantic"):
    mcp_server_main = importlib.reload(mcp_server_main)
from bedesten_mcp_module.models import (
    BedestenDecisionEntry,
    BedestenDocumentMarkdown,
    BedestenItemType,
    BedestenSearchData,
    BedestenSearchDataResponse,
    BedestenSearchRequest,
    BedestenSearchResponse,
)


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


class RelevantAfterFirstTenBedestenClient:
    def __init__(self):
        self.search_requests = []
        self.fetched_ids = []

    async def search_documents(self, search_request):
        self.search_requests.append(search_request)
        requested_size = search_request.data.pageSize
        return BedestenSearchResponse(
            data=BedestenSearchDataResponse(
                emsalKararList=[
                    decision("relevant" if index == 15 else f"doc-{index}")
                    for index in range(requested_size)
                ],
                total=100,
                start=0,
            ),
            metadata={},
        )

    async def get_document_as_markdown(self, document_id: str):
        self.fetched_ids.append(document_id)
        if document_id == "relevant":
            markdown = (
                "Alfa işlemi beta zararı hakkında doğrudan ilgili karar. "
                "Mahkeme alfa işlemi sonrasında doğan beta zararını ve tazminat "
                "sorumluluğunu ayrıntılı olarak değerlendirmiştir."
            )
        else:
            markdown = (
                "Bu karar kira sözleşmesinin uyarlanması ve temerrüt nedeniyle "
                "tahliye istemine ilişkindir. Uyuşmazlıkta kira bedelinin tespiti "
                "ve sözleşmenin sona ermesi tartışılmıştır."
            )
        return BedestenDocumentMarkdown(
            documentId=document_id,
            markdown_content=markdown,
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


class CaptureHttpClient:
    def __init__(self):
        self.payloads = []

    async def post(self, endpoint, json):
        self.payloads.append(json)
        return EmptySearchResponse()

    async def aclose(self):
        return None


class EmptySearchResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": {
                "emsalKararList": [],
                "total": 0,
                "start": 0,
            },
            "metadata": {},
        }


class BedestenSemanticToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_search_uses_max_candidates_for_metadata_window(self):
        original_client = mcp_server_main.bedesten_client_instance
        original_get_embedder = mcp_server_main.get_embedder
        client = RelevantAfterFirstTenBedestenClient()
        mcp_server_main.bedesten_client_instance = client
        mcp_server_main.get_embedder = lambda: KeywordEmbedder()
        try:
            response = await mcp_server_main.search_bedesten_semantic.fn(
                initial_keyword='"alfa işlemi"',
                query="Alfa işlemi sonrası beta zararı için tazminat sorumluluğu",
                court_types=["YARGITAYKARARI"],
                top_k=1,
                max_candidates=20,
                allow_broad_search=False,
                karar_yil_start="2020",
                karar_yil_end="2024",
            )
        finally:
            mcp_server_main.bedesten_client_instance = original_client
            mcp_server_main.get_embedder = original_get_embedder

        self.assertEqual(response["status"], "success")
        self.assertEqual(client.search_requests[0].data.pageSize, 20)
        self.assertEqual(client.search_requests[0].data.kararTarihiStart, "2020-01-01T00:00:00.000Z")
        self.assertEqual(client.search_requests[0].data.kararTarihiEnd, "2024-12-31T23:59:59.999Z")
        self.assertEqual(client.search_requests[0].data.sortFields, [])
        self.assertEqual(client.search_requests[0].data.sortDirection, "")
        self.assertIn("relevant", client.fetched_ids)
        self.assertLessEqual(len(client.fetched_ids), 20)
        self.assertEqual(response["results"][0]["document_id"], "relevant")

    async def test_semantic_search_rejects_invalid_year_range_before_searching(self):
        original_client = mcp_server_main.bedesten_client_instance
        client = RelevantAfterFirstTenBedestenClient()
        mcp_server_main.bedesten_client_instance = client
        try:
            response = await mcp_server_main.search_bedesten_semantic.fn(
                initial_keyword='"alfa işlemi"',
                query="Alfa işlemi sonrası beta zararı için tazminat sorumluluğu",
                court_types=["YARGITAYKARARI"],
                top_k=1,
                max_candidates=20,
                allow_broad_search=False,
                karar_yil_start="2025",
                karar_yil_end="2020",
            )
        finally:
            mcp_server_main.bedesten_client_instance = original_client

        self.assertEqual(response["status"], "validation_error")
        self.assertEqual(client.search_requests, [])

    async def test_semantic_search_default_court_types_include_local_civil_without_broad_flag(self):
        original_client = mcp_server_main.bedesten_client_instance
        original_get_embedder = mcp_server_main.get_embedder
        client = RelevantAfterFirstTenBedestenClient()
        mcp_server_main.bedesten_client_instance = client
        mcp_server_main.get_embedder = lambda: KeywordEmbedder()
        try:
            response = await mcp_server_main.search_bedesten_semantic.fn(
                initial_keyword='"alfa işlemi"',
                query="Alfa işlemi sonrası beta zararı için tazminat sorumluluğu",
                top_k=1,
                max_candidates=20,
                allow_broad_search=False,
            )
        finally:
            mcp_server_main.bedesten_client_instance = original_client
            mcp_server_main.get_embedder = original_get_embedder

        self.assertEqual(response["status"], "success")
        requested_court_types = [
            request.data.itemTypeList[0]
            for request in client.search_requests
        ]
        self.assertEqual(
            requested_court_types,
            ["YARGITAYKARARI", "ISTINAFHUKUK", "YERELHUKUK"],
        )

    async def test_semantic_search_explicit_three_court_override_still_requires_broad_flag(self):
        original_client = mcp_server_main.bedesten_client_instance
        client = RelevantAfterFirstTenBedestenClient()
        mcp_server_main.bedesten_client_instance = client
        try:
            response = await mcp_server_main.search_bedesten_semantic.fn(
                initial_keyword='"alfa işlemi"',
                query="Alfa işlemi sonrası beta zararı için tazminat sorumluluğu",
                court_types=["YARGITAYKARARI", "DANISTAYKARAR", "KYB"],
                top_k=1,
                max_candidates=20,
                allow_broad_search=False,
            )
        finally:
            mcp_server_main.bedesten_client_instance = original_client

        self.assertEqual(response["status"], "validation_error")
        self.assertEqual(client.search_requests, [])

    async def test_semantic_search_allows_explicit_default_court_scope_without_broad_flag(self):
        original_client = mcp_server_main.bedesten_client_instance
        original_get_embedder = mcp_server_main.get_embedder
        client = RelevantAfterFirstTenBedestenClient()
        mcp_server_main.bedesten_client_instance = client
        mcp_server_main.get_embedder = lambda: KeywordEmbedder()
        try:
            response = await mcp_server_main.search_bedesten_semantic.fn(
                initial_keyword='"alfa işlemi"',
                query="Alfa işlemi sonrası beta zararı için tazminat sorumluluğu",
                court_types=["YARGITAYKARARI", "ISTINAFHUKUK", "YERELHUKUK"],
                top_k=1,
                max_candidates=20,
                allow_broad_search=False,
            )
        finally:
            mcp_server_main.bedesten_client_instance = original_client
            mcp_server_main.get_embedder = original_get_embedder

        self.assertEqual(response["status"], "success")
        requested_court_types = [
            request.data.itemTypeList[0]
            for request in client.search_requests
        ]
        self.assertEqual(
            requested_court_types,
            ["YARGITAYKARARI", "ISTINAFHUKUK", "YERELHUKUK"],
        )

    async def test_default_sort_fields_are_omitted_from_bedesten_payload(self):
        from bedesten_mcp_module.client import BedestenApiClient

        client = BedestenApiClient()
        await client.http_client.aclose()
        capture_client = CaptureHttpClient()
        client.http_client = capture_client

        try:
            await client.search_documents(
                BedestenSearchRequest(
                    data=BedestenSearchData(
                        phrase='"alfa işlemi"',
                        itemTypeList=["YARGITAYKARARI"],
                        pageSize=5,
                        pageNumber=1,
                    )
                )
            )
        finally:
            await client.close_client_session()

        payload = capture_client.payloads[0]["data"]
        self.assertNotIn("sortFields", payload)
        self.assertNotIn("sortDirection", payload)


if __name__ == "__main__":
    unittest.main()
