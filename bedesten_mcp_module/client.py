# bedesten_mcp_module/client.py

import asyncio
import base64
import hashlib
import importlib
import io
import json
import logging
import os
import time
from typing import Optional

import httpx
from markitdown import MarkItDown

from .models import (
    BedestenSearchRequest, BedestenSearchResponse,
    BedestenDocumentRequest, BedestenDocumentResponse,
    BedestenDocumentMarkdown, BedestenDocumentRequestData
)
from .enums import get_full_birim_adi

logger = logging.getLogger(__name__)


class BedestenRateLimited(Exception):
    """Raised when the local rate-limit bucket would block longer than allowed.

    Carries the suggested retry-after (seconds) so callers can surface a
    structured 429-style response to the MCP client instead of silently
    blocking the event-loop slot for the full bucket-pause window.
    """

    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"local bucket would block {retry_after:.1f}s")


class _TokenBucket:
    """Asyncio token bucket with explicit back-pressure.

    Measured Bedesten limit (per source IP, 2026-05-08): 10 requests per
    rolling 30s window with full refill — equivalent to capacity=10,
    refill_rate=1 token / 3s. Even with margin, 429s still leak through
    when other clients share the egress IP, so we also expose
    ``penalize_until`` so callers can freeze the bucket when the server
    actually returns 429 (Retry-After).
    """

    def __init__(self, capacity: int, refill_per_s: float) -> None:
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._not_before = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, max_wait: Optional[float] = None) -> None:
        """Acquire one token. If ``max_wait`` is set and the next wait would
        exceed it, raise :class:`BedestenRateLimited` immediately instead of
        sleeping — keeps a single rate-limited request from holding the
        worker-slot for the full bucket-pause window (up to ~30s on 429)."""
        deadline = (time.monotonic() + max_wait) if max_wait is not None else None
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._not_before:
                    wait_s = self._not_before - now
                else:
                    self._tokens = min(
                        self.capacity,
                        self._tokens + (now - self._last) * self.refill_per_s,
                    )
                    self._last = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait_s = (1.0 - self._tokens) / self.refill_per_s
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if wait_s > remaining:
                    raise BedestenRateLimited(retry_after=wait_s)
            await asyncio.sleep(wait_s)

    def penalize_until(self, monotonic_deadline: float) -> None:
        """Pause the bucket until ``monotonic_deadline`` (drains tokens)."""
        self._not_before = max(self._not_before, monotonic_deadline)
        self._tokens = 0.0
        self._last = time.monotonic()


class _UpstashRedisClient:
    """Small async Upstash REST client used for Bedesten coordination.

    The project already depends on httpx, so this avoids adding a Redis package
    just for a few atomic commands.
    """

    def __init__(self, url: str, token: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=float(os.getenv("BEDESTEN_REDIS_TIMEOUT_S", "5.0")),
        )

    async def command(self, *parts):
        response = await self._client.post("/", json=list(parts))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload and payload["error"]:
            raise RuntimeError(f"Redis command failed: {payload['error']}")
        return payload.get("result")

    async def get(self, key: str) -> Optional[str]:
        result = await self.command("GET", key)
        if result is None:
            return None
        return str(result)

    async def set_json(self, key: str, payload: dict, ttl_s: int) -> None:
        value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self.command("SET", key, value, "EX", str(ttl_s))

    async def close(self) -> None:
        await self._client.aclose()


class _RedisProtocolClient:
    """Async Redis client for standard redis:// URLs."""

    def __init__(self, url: str) -> None:
        try:
            redis_asyncio = importlib.import_module("redis.asyncio")
        except ImportError as e:
            raise ImportError(
                "redis package is required for BEDESTEN_REDIS_URL/REDIS_URL. "
                "Install project dependencies or run: pip install redis"
            ) from e

        timeout = float(os.getenv("BEDESTEN_REDIS_TIMEOUT_S", "5.0"))
        self._client = redis_asyncio.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )

    async def command(self, *parts):
        return await self._client.execute_command(*parts)

    async def get(self, key: str) -> Optional[str]:
        result = await self._client.get(key)
        if result is None:
            return None
        if isinstance(result, bytes):
            return result.decode("utf-8")
        return str(result)

    async def set_json(self, key: str, payload: dict, ttl_s: int) -> None:
        value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self._client.set(key, value, ex=ttl_s)

    async def close(self) -> None:
        if hasattr(self._client, "aclose"):
            await self._client.aclose()
        else:
            await self._client.close()


_REDIS_TOKEN_BUCKET_SCRIPT = """
local bucket_key = KEYS[1]
local pause_key = KEYS[2]
local now_ms = tonumber(ARGV[1])
local capacity_scaled = tonumber(ARGV[2])
local refill_ms_per_token = tonumber(ARGV[3])
local scale = tonumber(ARGV[4])
local ttl_ms = tonumber(ARGV[5])

local paused_until = tonumber(redis.call("GET", pause_key) or "0")
if now_ms < paused_until then
  return {0, paused_until - now_ms}
end

local bucket = redis.call("HMGET", bucket_key, "tokens", "updated_ms")
local tokens = tonumber(bucket[1]) or capacity_scaled
local updated_ms = tonumber(bucket[2]) or now_ms

if now_ms > updated_ms then
  local refill = math.floor(((now_ms - updated_ms) * scale) / refill_ms_per_token)
  tokens = math.min(capacity_scaled, tokens + refill)
  updated_ms = now_ms
end

if tokens >= scale then
  tokens = tokens - scale
  redis.call("HSET", bucket_key, "tokens", tokens, "updated_ms", updated_ms)
  redis.call("PEXPIRE", bucket_key, ttl_ms)
  return {1, 0}
end

redis.call("HSET", bucket_key, "tokens", tokens, "updated_ms", updated_ms)
redis.call("PEXPIRE", bucket_key, ttl_ms)
local wait_ms = math.ceil(((scale - tokens) * refill_ms_per_token) / scale)
return {0, wait_ms}
"""


_REDIS_PAUSE_SCRIPT = """
local pause_key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local deadline_ms = tonumber(ARGV[2])
local ttl_ms = math.max(1, deadline_ms - now_ms)
local current = tonumber(redis.call("GET", pause_key) or "0")
if deadline_ms > current then
  redis.call("PSETEX", pause_key, ttl_ms, deadline_ms)
  return deadline_ms
end
return current
"""


class _RedisTokenBucket:
    """Redis-backed token bucket with in-process fallback for Redis outages."""

    _SCALE = 1000

    def __init__(
        self,
        redis_client,
        key_prefix: str,
        capacity: int,
        refill_s: float,
    ) -> None:
        self._redis = redis_client
        self._bucket_key = f"{key_prefix}:rate:bucket"
        self._pause_key = f"{key_prefix}:rate:paused_until_ms"
        self._capacity_scaled = int(capacity * self._SCALE)
        self._refill_ms = max(1, int(refill_s * 1000))
        self._ttl_ms = max(60000, int(max(capacity, 1) * refill_s * 4000))
        self._fallback = _TokenBucket(capacity=capacity, refill_per_s=1.0 / refill_s)

    async def acquire(self, max_wait: Optional[float] = None) -> None:
        deadline = (time.monotonic() + max_wait) if max_wait is not None else None
        while True:
            try:
                now_ms = int(time.time() * 1000)
                result = await self._redis.command(
                    "EVAL",
                    _REDIS_TOKEN_BUCKET_SCRIPT,
                    "2",
                    self._bucket_key,
                    self._pause_key,
                    str(now_ms),
                    str(self._capacity_scaled),
                    str(self._refill_ms),
                    str(self._SCALE),
                    str(self._ttl_ms),
                )
                acquired = int(result[0]) == 1
                wait_s = int(result[1]) / 1000.0
            except Exception as e:
                logger.warning(
                    "Bedesten Redis rate limiter unavailable; using local fallback: %s",
                    e,
                )
                await self._fallback.acquire(max_wait=max_wait)
                return

            if acquired:
                return

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if wait_s > remaining:
                    raise BedestenRateLimited(retry_after=wait_s)
            await asyncio.sleep(max(wait_s, 0.001))

    async def penalize_for(self, retry_after_s: float) -> None:
        deadline_ms = int((time.time() + retry_after_s) * 1000)
        try:
            await self._redis.command(
                "EVAL",
                _REDIS_PAUSE_SCRIPT,
                "1",
                self._pause_key,
                str(int(time.time() * 1000)),
                str(deadline_ms),
            )
        except Exception as e:
            logger.warning(
                "Bedesten Redis pause update failed; using local fallback: %s",
                e,
            )
            self._fallback.penalize_until(time.monotonic() + retry_after_s)


class BedestenApiClient:
    """
    API Client for Bedesten (bedesten.adalet.gov.tr) - Alternative legal decision search system.
    Currently used for Yargıtay decisions, but can be extended for other court types.
    """
    BASE_URL = "https://bedesten.adalet.gov.tr"
    SEARCH_ENDPOINT = "/emsal-karar/searchDocuments"
    DOCUMENT_ENDPOINT = "/emsal-karar/getDocumentContent"
    
    # Measured limit (per source IP): 10 requests per 30s window with full
    # refill (≈ 1 token / 3s steady). We default to 1-token capacity and
    # 3.5s spacing (no burst, ~14% safety margin). Override via env:
    #   BEDESTEN_RATE_CAPACITY (default 1)
    #   BEDESTEN_RATE_REFILL_S (default 3.5; seconds per token)
    #   BEDESTEN_RATE_MAX_WAIT_S (default 8.0; max seconds to wait in the
    #     local bucket before returning a structured 429 to the caller)
    _DEFAULT_CAPACITY = int(os.getenv("BEDESTEN_RATE_CAPACITY", "1"))
    _DEFAULT_REFILL_S = float(os.getenv("BEDESTEN_RATE_REFILL_S", "3.5"))
    _DEFAULT_MAX_WAIT_S = float(os.getenv("BEDESTEN_RATE_MAX_WAIT_S", "8.0"))

    def __init__(self, request_timeout: float = 60.0):
        self._capacity = int(os.getenv("BEDESTEN_RATE_CAPACITY", str(self._DEFAULT_CAPACITY)))
        self._refill_s = float(os.getenv("BEDESTEN_RATE_REFILL_S", str(self._DEFAULT_REFILL_S)))
        self._max_wait_s = float(os.getenv("BEDESTEN_RATE_MAX_WAIT_S", str(self._DEFAULT_MAX_WAIT_S)))
        self._rate_backend = os.getenv("BEDESTEN_RATE_BACKEND", "memory").strip().lower()
        self._cache_backend = os.getenv(
            "BEDESTEN_CACHE_BACKEND",
            "redis" if self._rate_backend == "redis" else "off",
        ).strip().lower()
        self._redis_prefix = os.getenv("BEDESTEN_REDIS_PREFIX", "yargi-mcp:bedesten")
        self._search_cache_ttl_s = int(os.getenv("BEDESTEN_SEARCH_CACHE_TTL_S", "21600"))
        self._document_cache_ttl_s = int(os.getenv("BEDESTEN_DOCUMENT_CACHE_TTL_S", "2592000"))

        self.http_client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Accept": "*/*",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "AdaletApplicationName": "UyapMevzuat",
                "Content-Type": "application/json; charset=utf-8",
                "Origin": "https://mevzuat.adalet.gov.tr",
                "Referer": "https://mevzuat.adalet.gov.tr/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
            },
            timeout=request_timeout
        )
        self._redis = None
        self._cache_enabled = False

        if self._rate_backend == "redis" or self._cache_backend == "redis":
            redis_url = os.getenv("BEDESTEN_REDIS_URL") or os.getenv("REDIS_URL")
            redis_rest_url = (
                os.getenv("BEDESTEN_REDIS_REST_URL")
                or os.getenv("UPSTASH_REDIS_REST_URL")
            )
            redis_rest_token = (
                os.getenv("BEDESTEN_REDIS_REST_TOKEN")
                or os.getenv("UPSTASH_REDIS_REST_TOKEN")
            )

            if redis_url:
                self._redis = _RedisProtocolClient(redis_url)
            elif redis_rest_url and redis_rest_token:
                self._redis = _UpstashRedisClient(redis_rest_url, redis_rest_token)
            else:
                raise ValueError(
                    "BEDESTEN_RATE_BACKEND=redis or BEDESTEN_CACHE_BACKEND=redis "
                    "requires BEDESTEN_REDIS_URL/REDIS_URL for standard Redis, "
                    "or BEDESTEN_REDIS_REST_URL/BEDESTEN_REDIS_REST_TOKEN "
                    "or UPSTASH_REDIS_REST_URL/UPSTASH_REDIS_REST_TOKEN for "
                    "Upstash REST Redis."
                )

        if self._rate_backend == "redis":
            self._bucket = _RedisTokenBucket(
                redis_client=self._redis,
                key_prefix=self._redis_prefix,
                capacity=self._capacity,
                refill_s=self._refill_s,
            )
        else:
            self._bucket = _TokenBucket(
                capacity=self._capacity,
                refill_per_s=1.0 / self._refill_s,
            )

        self._cache_enabled = self._cache_backend == "redis" and self._redis is not None

    def _cache_key(self, kind: str, payload: object) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{self._redis_prefix}:cache:{kind}:{digest}"

    async def _cache_get_json(self, key: str) -> Optional[dict]:
        if not self._cache_enabled:
            return None
        try:
            cached = await self._redis.get(key)
            if not cached:
                return None
            return json.loads(cached)
        except Exception as e:
            logger.warning("Bedesten Redis cache read failed for %s: %s", key, e)
            return None

    async def _cache_set_json(self, key: str, payload: dict, ttl_s: int) -> None:
        if not self._cache_enabled or ttl_s <= 0:
            return
        try:
            await self._redis.set_json(key, payload, ttl_s)
        except Exception as e:
            logger.warning("Bedesten Redis cache write failed for %s: %s", key, e)

    async def _handle_429(self, response: httpx.Response, op: str) -> None:
        """Apply back-pressure to the shared bucket based on Retry-After."""
        retry_after_raw = response.headers.get("Retry-After", "")
        try:
            retry_after = float(retry_after_raw)
        except (TypeError, ValueError):
            retry_after = 30.0
        # Cap penalty so a hostile/buggy server can't freeze us indefinitely.
        retry_after = max(1.0, min(retry_after, 60.0))
        retry_after_with_margin = retry_after + 0.5
        if hasattr(self._bucket, "penalize_for"):
            await self._bucket.penalize_for(retry_after_with_margin)
        else:
            self._bucket.penalize_until(time.monotonic() + retry_after_with_margin)
        logger.warning(
            f"BedestenApiClient: 429 on {op}; bucket paused {retry_after_with_margin:.1f}s"
        )
    
    async def search_documents(self, search_request: BedestenSearchRequest) -> BedestenSearchResponse:
        """
        Search for documents using Bedesten API.
        Currently supports: YARGITAYKARARI, DANISTAYKARARI, YERELHUKMAHKARARI, etc.
        """
        logger.info(f"BedestenApiClient: Searching documents with phrase: {search_request.data.phrase}")
        
        # Map abbreviated birimAdi to full Turkish name before sending to API
        original_birim_adi = search_request.data.birimAdi
        mapped_birim_adi = get_full_birim_adi(original_birim_adi)
        search_request.data.birimAdi = mapped_birim_adi
        if original_birim_adi != "ALL":
            logger.info(f"BedestenApiClient: Mapped birimAdi '{original_birim_adi}' to '{mapped_birim_adi}'")
        
        try:
            # Create request dict and remove birimAdi if empty
            request_dict = search_request.model_dump()
            if not request_dict["data"]["birimAdi"]:  # Remove if empty string
                del request_dict["data"]["birimAdi"]

            cache_key = self._cache_key("search", request_dict)
            cached_response = await self._cache_get_json(cache_key)
            if cached_response is not None:
                logger.info("BedestenApiClient: Search cache hit")
                return BedestenSearchResponse(**cached_response)
            
            await self._bucket.acquire(max_wait=self._max_wait_s)
            response = await self.http_client.post(
                self.SEARCH_ENDPOINT,
                json=request_dict
            )
            if response.status_code == 429:
                await self._handle_429(response, "search")
            response.raise_for_status()
            response_json = response.json()
            await self._cache_set_json(cache_key, response_json, self._search_cache_ttl_s)

            # Parse and return the response
            return BedestenSearchResponse(**response_json)

        except httpx.RequestError as e:
            logger.error(f"BedestenApiClient: HTTP request error during search: {e}")
            raise
        except Exception as e:
            logger.error(f"BedestenApiClient: Error processing search response: {e}")
            raise
    
    async def get_document_as_markdown(self, document_id: str) -> BedestenDocumentMarkdown:
        """
        Get document content and convert to markdown.
        Handles both HTML (text/html) and PDF (application/pdf) content types.
        """
        logger.info(f"BedestenApiClient: Fetching document for markdown conversion (ID: {document_id})")
        
        try:
            cache_key = self._cache_key("document", {"documentId": document_id})
            cached_document = await self._cache_get_json(cache_key)
            if cached_document is not None:
                logger.info("BedestenApiClient: Document cache hit (ID: %s)", document_id)
                return BedestenDocumentMarkdown(**cached_document)

            # Prepare request
            doc_request = BedestenDocumentRequest(
                data=BedestenDocumentRequestData(documentId=document_id)
            )
            
            # Get document
            await self._bucket.acquire(max_wait=self._max_wait_s)
            response = await self.http_client.post(
                self.DOCUMENT_ENDPOINT,
                json=doc_request.model_dump()
            )
            if response.status_code == 429:
                await self._handle_429(response, f"document {document_id}")
            response.raise_for_status()
            response_json = response.json()
            doc_response = BedestenDocumentResponse(**response_json)
            
            # Add null safety checks for document data
            if not hasattr(doc_response, 'data') or doc_response.data is None:
                raise ValueError("Document response does not contain data")
            
            if not hasattr(doc_response.data, 'content') or doc_response.data.content is None:
                raise ValueError("Document data does not contain content")
                
            if not hasattr(doc_response.data, 'mimeType') or doc_response.data.mimeType is None:
                raise ValueError("Document data does not contain mimeType")
            
            # Decode base64 content with error handling
            try:
                content_bytes = base64.b64decode(doc_response.data.content)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 content: {str(e)}")
            
            mime_type = doc_response.data.mimeType
            
            logger.info(f"BedestenApiClient: Document mime type: {mime_type}")
            
            # Convert to markdown based on mime type. markitdown is sync and
            # PDF parsing in particular can block the event-loop for seconds,
            # which on a single-worker uvicorn deployment stalls every other
            # in-flight MCP request and new TLS handshakes. Offload to a
            # thread so the event-loop stays responsive.
            if mime_type == "text/html":
                html_content = content_bytes.decode('utf-8')
                markdown_content = await asyncio.to_thread(
                    self._convert_html_to_markdown, html_content
                )
            elif mime_type == "application/pdf":
                markdown_content = await asyncio.to_thread(
                    self._convert_pdf_to_markdown, content_bytes
                )
            else:
                logger.warning(f"Unsupported mime type: {mime_type}")
                markdown_content = f"Unsupported content type: {mime_type}. Unable to convert to markdown."
            
            document_markdown = BedestenDocumentMarkdown(
                documentId=document_id,
                markdown_content=markdown_content,
                source_url=f"https://mevzuat.adalet.gov.tr/ictihat/{document_id}",
                mime_type=mime_type
            )
            await self._cache_set_json(
                cache_key,
                document_markdown.model_dump(),
                self._document_cache_ttl_s,
            )
            return document_markdown
            
        except httpx.RequestError as e:
            logger.error(f"BedestenApiClient: HTTP error fetching document {document_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"BedestenApiClient: Error processing document {document_id}: {e}")
            raise
    
    def _convert_html_to_markdown(self, html_content: str) -> Optional[str]:
        """Convert HTML to Markdown using MarkItDown"""
        if not html_content:
            return None
            
        try:
            # Convert HTML string to bytes and create BytesIO stream
            html_bytes = html_content.encode('utf-8')
            html_stream = io.BytesIO(html_bytes)
            
            # Pass BytesIO stream to MarkItDown to avoid temp file creation
            md_converter = MarkItDown()
            result = md_converter.convert(html_stream)
            markdown_content = result.text_content
            
            logger.info("Successfully converted HTML to Markdown")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {e}")
            return f"Error converting HTML content: {str(e)}"
    
    def _convert_pdf_to_markdown(self, pdf_bytes: bytes) -> Optional[str]:
        """Convert PDF to Markdown using MarkItDown"""
        if not pdf_bytes:
            return None
            
        try:
            # Create BytesIO stream from PDF bytes
            pdf_stream = io.BytesIO(pdf_bytes)
            
            # Pass BytesIO stream to MarkItDown to avoid temp file creation
            md_converter = MarkItDown()
            result = md_converter.convert(pdf_stream)
            markdown_content = result.text_content
            
            logger.info("Successfully converted PDF to Markdown")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Error converting PDF to Markdown: {e}")
            return f"Error converting PDF content: {str(e)}. The document may be corrupted or in an unsupported format."
    
    async def close_client_session(self):
        """Close HTTP client session"""
        await self.http_client.aclose()
        if self._redis is not None:
            await self._redis.close()
        logger.info("BedestenApiClient: HTTP client session closed.")
