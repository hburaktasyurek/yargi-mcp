# semantic_search/deep_bedesten.py

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from bedesten_mcp_module.client import BedestenRateLimited
from bedesten_mcp_module.models import (
    BedestenCourtTypeEnum,
    BedestenSearchData,
    BedestenSearchRequest,
)
from semantic_search.processor import DocumentProcessor

logger = logging.getLogger(__name__)

ALLOWED_COURT_TYPES = {
    "YARGITAYKARARI",
    "ISTINAFHUKUK",
    "DANISTAYKARAR",
    "YERELHUKUK",
    "KYB",
}

BEDESTEN_DOCUMENT_SOURCE_URL_TEMPLATE = (
    "https://bedesten.adalet.gov.tr/emsal-karar/getDocumentContent?documentId={document_id}"
)
RESULT_CHUNK_PREVIEW_CHARS = 700
_LEXICON_CACHE: Dict[str, Tuple[float, Optional[List[Dict[str, Any]]]]] = {}

LEGAL_EXPANSION_PROFILES = [
    {
        "name": "muris_muvazaasi",
        "triggers": ["muris", "muvazaa", "mal kaçırma", "mirasçılardan"],
        "terms": [
            "muris muvazaası",
            "tapu iptali ve tescil",
            "mirasçılardan mal kaçırma",
            "gizli bağış",
            "görünürde satış",
        ],
    },
    {
        "name": "trafik_sigorta",
        "triggers": ["trafik kazası", "değer kaybı", "sigorta", "zmss"],
        "terms": [
            "araç değer kaybı",
            "trafik kazası",
            "zorunlu mali sorumluluk sigortası",
            "sigorta şirketinin sorumluluğu",
            "eksper raporu",
        ],
    },
    {
        "name": "is_hukuku",
        "triggers": ["işçi", "işveren", "kıdem", "ihbar", "fazla mesai"],
        "terms": [
            "kıdem tazminatı",
            "ihbar tazminatı",
            "fazla çalışma ücreti",
            "iş sözleşmesinin feshi",
            "işçilik alacağı",
        ],
    },
    {
        "name": "kira_tahliye",
        "triggers": ["kira", "kiracı", "tahliye", "kira bedeli"],
        "terms": [
            "tahliye davası",
            "kira bedelinin tespiti",
            "temerrüt nedeniyle tahliye",
            "ihtiyaç nedeniyle tahliye",
            "kira sözleşmesi",
        ],
    },
]


def load_legal_expansion_profiles() -> Optional[List[Dict[str, Any]]]:
    """Load optional expansion profiles from BEDESTEN_DEEP_LEXICON_PATH.

    The JSON file must contain a list of objects with ``name``, ``triggers``
    and ``terms``. Invalid files are ignored so the MCP tool remains usable.
    """
    path = os.getenv("BEDESTEN_DEEP_LEXICON_PATH", "").strip()
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError as e:
        logger.warning("Could not stat Bedesten deep lexicon at %s: %s", path, e)
        return None
    cached = _LEXICON_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as e:
        logger.warning("Could not load Bedesten deep lexicon from %s: %s", path, e)
        return None

    if not isinstance(payload, list):
        logger.warning("Bedesten deep lexicon must be a JSON list: %s", path)
        return None

    profiles: List[Dict[str, Any]] = []
    for raw_profile in payload:
        if not isinstance(raw_profile, dict):
            continue
        name = str(raw_profile.get("name", "")).strip()
        triggers = raw_profile.get("triggers")
        terms = raw_profile.get("terms")
        if not name or not isinstance(triggers, list) or not isinstance(terms, list):
            continue
        normalized_triggers = [str(trigger).strip().lower() for trigger in triggers if str(trigger).strip()]
        normalized_terms = [str(term).strip() for term in terms if str(term).strip()]
        if normalized_triggers and normalized_terms:
            profiles.append(
                {
                    "name": name,
                    "triggers": normalized_triggers,
                    "terms": normalized_terms,
                }
            )

    loaded_profiles = profiles or None
    _LEXICON_CACHE[path] = (mtime, loaded_profiles)
    return loaded_profiles


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _quote(term: str) -> str:
    if " " in term and not (term.startswith('"') and term.endswith('"')):
        return f'"{term}"'
    return term


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        key = _normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def generate_bedesten_deep_queries(
    question: str,
    seed_terms: Optional[List[str]] = None,
    max_queries: int = 8,
    expansion_profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create controlled legal query expansion without letting an LLM roam.

    The output is intentionally plain: the MCP tool can show generated queries
    to users, and tests can verify behavior through this public function.
    """
    seed_terms = seed_terms or []
    question_normalized = _normalize_text(question)
    matched_profiles = []
    expansion_terms: List[str] = []

    profiles = expansion_profiles if expansion_profiles is not None else LEGAL_EXPANSION_PROFILES
    for profile in profiles:
        if any(trigger in question_normalized for trigger in profile["triggers"]):
            matched_profiles.append(profile["name"])
            expansion_terms.extend(profile["terms"])

    expansion_terms = _dedupe_preserve_order(seed_terms + expansion_terms)
    short_question_terms = [
        token
        for token in re.findall(r"[\wÇĞIİÖŞÜçğıiöşü]+", question)
        if len(token) >= 4
    ][:3]

    queries: List[str] = []
    for term in expansion_terms:
        queries.append(_quote(term))

    if len(expansion_terms) >= 2:
        main = _quote(expansion_terms[0])
        for term in expansion_terms[1:5]:
            queries.append(f"{main} AND {_quote(term)}")

    if short_question_terms and expansion_terms:
        anchor = _quote(expansion_terms[0])
        queries.append(f"{anchor} AND {short_question_terms[0]}")

    if not queries:
        queries.append(question.strip())

    return {
        "matched_profiles": matched_profiles,
        "terms": expansion_terms,
        "queries": _dedupe_preserve_order(queries)[:max_queries],
    }


def _court_type_value(court_type: Any) -> str:
    return getattr(court_type, "value", str(court_type))


def _bedesten_document_source_url(document_id: Optional[str]) -> Optional[str]:
    if not document_id:
        return None
    return BEDESTEN_DOCUMENT_SOURCE_URL_TEMPLATE.format(document_id=document_id)


def _decision_metadata(decision: Any) -> Dict[str, Any]:
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
        "court_type": court_type,
        "birim_adi": getattr(decision, "birimAdi", None),
        "esas_no": getattr(decision, "esasNo", None),
        "karar_no": getattr(decision, "kararNo", None),
        "karar_tarihi": getattr(decision, "kararTarihiStr", None),
        "title": " - ".join(title_parts) if title_parts else f"Document {document_id}",
        "source_url": _bedesten_document_source_url(document_id),
    }


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / norm if norm > 0 else embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def _semantic_response(
    status: str,
    message: str,
    start_time: float,
    diagnostics: Dict[str, Any],
    **extra: Any,
) -> Dict[str, Any]:
    diagnostics["total_ms"] = int((time.monotonic() - start_time) * 1000)
    response = {
        "status": status,
        "message": message,
        "diagnostics": diagnostics,
    }
    response.update(extra)
    return response


async def search_bedesten_deep_semantic(
    *,
    bedesten_client: Any,
    embedder: Any,
    question: str,
    court_type: BedestenCourtTypeEnum,
    seed_terms: Optional[List[str]] = None,
    max_queries: int = 8,
    max_search_results: int = 50,
    max_fulltext_fetches: int = 20,
    top_k: int = 8,
    use_expansion: bool = True,
) -> Dict[str, Any]:
    """Search one Bedesten court type deeply, then rank fetched chunks semantically.

    This is deliberately bounded and sequential so Bedesten rate limits remain
    the controlling constraint instead of local concurrency.
    """
    start_time = time.monotonic()
    court_type_value = _court_type_value(court_type)
    diagnostics: Dict[str, Any] = {
        "candidate_count": 0,
        "deduped_candidate_count": 0,
        "searched_query_count": 0,
        "fetched_count": 0,
        "failed_fetches": 0,
        "chunk_count": 0,
        "embedding_model": getattr(embedder, "model", None),
        "provider": getattr(embedder, "provider", None),
        "total_ms": 0,
    }

    if not question or len(question.strip()) < 3:
        return _semantic_response(
            "validation_error",
            "question must contain at least 3 non-whitespace characters.",
            start_time,
            diagnostics,
            court_type=court_type_value,
        )
    if isinstance(court_type, (list, tuple, set)) or court_type_value not in ALLOWED_COURT_TYPES:
        return _semantic_response(
            "validation_error",
            "court_type must be exactly one Bedesten court type.",
            start_time,
            diagnostics,
            court_type=court_type_value,
        )

    max_queries = max(1, min(int(max_queries), 8))
    max_search_results = max(1, min(int(max_search_results), 50))
    max_fulltext_fetches = max(1, min(int(max_fulltext_fetches), 25))
    top_k = max(1, min(int(top_k), max_fulltext_fetches))

    expansion = (
        generate_bedesten_deep_queries(
            question,
            seed_terms=seed_terms,
            max_queries=max_queries,
            expansion_profiles=load_legal_expansion_profiles(),
        )
        if use_expansion
        else {"matched_profiles": [], "terms": seed_terms or [], "queries": [question.strip()]}
    )
    generated_queries = expansion["queries"][:max_queries]

    candidates_by_id: Dict[str, Dict[str, Any]] = {}
    for query in generated_queries:
        if len(candidates_by_id) >= max_search_results:
            break
        page_size = min(10, max_search_results - len(candidates_by_id))
        try:
            search_response = await bedesten_client.search_documents(
                BedestenSearchRequest(
                    data=BedestenSearchData(
                        pageSize=page_size,
                        pageNumber=1,
                        itemTypeList=[court_type_value],
                        phrase=query,
                    )
                )
            )
            diagnostics["searched_query_count"] += 1
        except (BedestenRateLimited, httpx.HTTPStatusError):
            raise
        except Exception as e:
            logger.warning("Deep semantic Bedesten search failed for %r: %s", query, e)
            continue

        decisions = []
        if search_response.data and search_response.data.emsalKararList:
            decisions = search_response.data.emsalKararList
        diagnostics["candidate_count"] += len(decisions)

        for decision in decisions:
            metadata = _decision_metadata(decision)
            document_id = metadata.get("document_id")
            if not document_id:
                continue
            candidate = candidates_by_id.setdefault(
                document_id,
                {
                    "metadata": metadata,
                    "matched_queries": [],
                },
            )
            candidate["matched_queries"].append(query)

    diagnostics["deduped_candidate_count"] = len(candidates_by_id)
    if not candidates_by_id:
        return _semantic_response(
            "no_results",
            "No candidate documents found in the selected court type.",
            start_time,
            diagnostics,
            court_type=court_type_value,
            generated_queries=generated_queries,
            expansion=expansion,
            results=[],
        )

    ranked_candidates = sorted(
        candidates_by_id.values(),
        key=lambda candidate: len(candidate["matched_queries"]),
        reverse=True,
    )[:max_fulltext_fetches]

    processor = DocumentProcessor(chunk_size=1200, chunk_overlap=250, min_chunk_size=80)
    chunk_texts: List[str] = []
    chunk_records: List[Dict[str, Any]] = []

    for candidate in ranked_candidates:
        document_id = candidate["metadata"]["document_id"]
        try:
            document = await bedesten_client.get_document_as_markdown(document_id)
        except (BedestenRateLimited, httpx.HTTPStatusError):
            raise
        except Exception as e:
            diagnostics["failed_fetches"] += 1
            logger.warning("Deep semantic document fetch failed for %s: %s", document_id, e)
            continue

        markdown = document.markdown_content or ""
        document_source_url = document.source_url or ""
        source_url = (
            document_source_url
            if document_source_url.startswith("https://bedesten.adalet.gov.tr/")
            else candidate["metadata"].get("source_url")
        )
        candidate_metadata = {
            **candidate["metadata"],
            "source_url": source_url,
        }
        chunks = processor.process_document(
            document_id=document_id,
            text=markdown,
            metadata=candidate_metadata.copy(),
        )
        if not chunks:
            diagnostics["failed_fetches"] += 1
            continue

        diagnostics["fetched_count"] += 1
        for chunk in chunks:
            chunk_texts.append(chunk.text)
            chunk_records.append(
                {
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": candidate_metadata,
                    "matched_queries": candidate["matched_queries"],
                }
            )

    diagnostics["chunk_count"] = len(chunk_records)
    if not chunk_records:
        return _semantic_response(
            "embedding_error",
            "No fetched document content could be chunked for semantic ranking.",
            start_time,
            diagnostics,
            court_type=court_type_value,
            generated_queries=generated_queries,
            expansion=expansion,
            results=[],
        )

    query_embedding = _normalize_embeddings(
        await asyncio.to_thread(embedder.encode_query, question, "legal issue retrieval")
    )
    chunk_embeddings = _normalize_embeddings(
        await asyncio.to_thread(embedder.encode_documents, chunk_texts)
    )
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    similarities = np.atleast_1d(np.dot(chunk_embeddings, query_embedding.T).squeeze())

    documents: Dict[str, Dict[str, Any]] = {}
    for index, score in enumerate(similarities):
        chunk_record = chunk_records[index]
        document_id = chunk_record["document_id"]
        document = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "score": 0.0,
                "metadata": chunk_record["metadata"],
                "matched_queries": chunk_record["matched_queries"],
                "best_chunks": [],
                "source_url": chunk_record["metadata"].get("source_url"),
            },
        )
        score_float = float(score)
        document["score"] = max(document["score"], score_float)
        document["best_chunks"].append(
            {
                "score": score_float,
                "chunk_index": chunk_record["chunk_index"],
                # Keep result payloads compact while chunks remain large enough for scoring.
                "text": chunk_record["text"][:RESULT_CHUNK_PREVIEW_CHARS],
            }
        )

    results = sorted(documents.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    for result in results:
        result["best_chunks"] = sorted(
            result["best_chunks"],
            key=lambda chunk: chunk["score"],
            reverse=True,
        )[:3]

    return _semantic_response(
        "success",
        "Deep semantic search completed in the selected court type.",
        start_time,
        diagnostics,
        court_type=court_type_value,
        generated_queries=generated_queries,
        expansion=expansion,
        results=results,
    )
