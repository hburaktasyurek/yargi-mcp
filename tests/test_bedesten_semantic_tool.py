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
    BedestenSearchDataResponse,
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
            )
        finally:
            mcp_server_main.bedesten_client_instance = original_client
            mcp_server_main.get_embedder = original_get_embedder

        self.assertEqual(response["status"], "success")
        self.assertEqual(client.search_requests[0].data.pageSize, 20)
        self.assertIn("relevant", client.fetched_ids)
        self.assertLessEqual(len(client.fetched_ids), 20)
        self.assertEqual(response["results"][0]["document_id"], "relevant")


if __name__ == "__main__":
    unittest.main()
