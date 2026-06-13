import json
import os
import sys
import types
import unittest
from contextlib import contextmanager

import httpx

from bedesten_mcp_module.client import BedestenApiClient, BedestenRateLimited
from bedesten_mcp_module.models import BedestenSearchData, BedestenSearchRequest


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


@contextmanager
def patched_redis_asyncio(fake_connection):
    previous_redis = sys.modules.get("redis")
    previous_asyncio = sys.modules.get("redis.asyncio")
    redis_module = types.ModuleType("redis")
    redis_asyncio_module = types.ModuleType("redis.asyncio")

    class Redis:
        @staticmethod
        def from_url(url, **kwargs):
            fake_connection.urls.append((url, kwargs))
            return fake_connection

    redis_asyncio_module.Redis = Redis
    redis_module.asyncio = redis_asyncio_module
    sys.modules["redis"] = redis_module
    sys.modules["redis.asyncio"] = redis_asyncio_module
    try:
        yield
    finally:
        if previous_redis is None:
            sys.modules.pop("redis", None)
        else:
            sys.modules["redis"] = previous_redis

        if previous_asyncio is None:
            sys.modules.pop("redis.asyncio", None)
        else:
            sys.modules["redis.asyncio"] = previous_asyncio


def sample_search_request() -> BedestenSearchRequest:
    return BedestenSearchRequest(
        data=BedestenSearchData(
            pageSize=1,
            pageNumber=1,
            itemTypeList=["YARGITAYKARARI"],
            phrase="mülkiyet",
        )
    )


def sample_search_response(total: int = 1) -> dict:
    return {
        "data": {
            "emsalKararList": [
                {
                    "documentId": "doc-1",
                    "itemType": {
                        "name": "YARGITAYKARARI",
                        "description": "Yargıtay Kararı",
                    },
                    "birimAdi": "1. Hukuk Dairesi",
                    "kararTarihi": "2024-01-01T00:00:00.000Z",
                    "kararTarihiStr": "01.01.2024",
                }
            ],
            "total": total,
            "start": 0,
        },
        "metadata": {},
    }


class FakeUpstashRedis:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.values = {}
        self.paused_until_ms = 0

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.fail:
                return httpx.Response(503, json={"error": "redis unavailable"})

            command = json.loads(request.content.decode("utf-8"))
            name = command[0].upper()

            if name == "GET":
                return httpx.Response(200, json={"result": self.values.get(command[1])})

            if name == "SET":
                self.values[command[1]] = command[2]
                return httpx.Response(200, json={"result": "OK"})

            if name == "EVAL" and command[2] == "1":
                deadline_ms = int(command[5])
                if deadline_ms > self.paused_until_ms:
                    self.paused_until_ms = deadline_ms
                    self.values[command[3]] = str(deadline_ms)
                return httpx.Response(200, json={"result": self.paused_until_ms})

            if name == "EVAL" and command[2] == "2":
                now_ms = int(command[5])
                if now_ms < self.paused_until_ms:
                    return httpx.Response(
                        200,
                        json={"result": [0, self.paused_until_ms - now_ms]},
                    )
                return httpx.Response(200, json={"result": [1, 0]})

            return httpx.Response(400, json={"error": f"unsupported command: {command}"})

        return httpx.MockTransport(handler)


class FakeProtocolRedisConnection:
    def __init__(self):
        self.urls = []
        self.values = {}
        self.paused_until_ms = 0

    async def execute_command(self, *command):
        name = command[0].upper()

        if name == "EVAL" and command[2] == "1":
            deadline_ms = int(command[5])
            if deadline_ms > self.paused_until_ms:
                self.paused_until_ms = deadline_ms
                self.values[command[3]] = str(deadline_ms)
            return self.paused_until_ms

        if name == "EVAL" and command[2] == "2":
            now_ms = int(command[5])
            if now_ms < self.paused_until_ms:
                return [0, self.paused_until_ms - now_ms]
            return [1, 0]

        raise AssertionError(f"unsupported command: {command}")

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value
        return True

    async def aclose(self):
        return None


async def attach_transports(
    client: BedestenApiClient,
    bedesten_transport: httpx.MockTransport,
    redis_transport: httpx.MockTransport | None = None,
) -> None:
    await client.http_client.aclose()
    client.http_client = httpx.AsyncClient(
        base_url=client.BASE_URL,
        transport=bedesten_transport,
    )
    if redis_transport is not None and client._redis is not None:
        await client._redis._client.aclose()
        client._redis._client = httpx.AsyncClient(
            base_url="https://redis.test",
            transport=redis_transport,
            headers={"Authorization": "Bearer test-token"},
        )


class BedestenRedisCoordinationTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_redis_url_can_coordinate_rate_limit_and_cache(self):
        calls = {"bedesten": 0}

        def bedesten_handler(request: httpx.Request) -> httpx.Response:
            calls["bedesten"] += 1
            return httpx.Response(200, json=sample_search_response(total=2))

        redis = FakeProtocolRedisConnection()
        with patched_env(
            BEDESTEN_RATE_BACKEND="redis",
            BEDESTEN_CACHE_BACKEND="redis",
            BEDESTEN_REDIS_URL="redis://localhost:6379/0",
            UPSTASH_REDIS_REST_URL=None,
            UPSTASH_REDIS_REST_TOKEN=None,
        ), patched_redis_asyncio(redis):
            client = BedestenApiClient()
            await attach_transports(client, httpx.MockTransport(bedesten_handler))
            try:
                first = await client.search_documents(sample_search_request())
                second = await client.search_documents(sample_search_request())
            finally:
                await client.close_client_session()

        self.assertEqual(redis.urls[0][0], "redis://localhost:6379/0")
        self.assertEqual(first.data.total, 2)
        self.assertEqual(second.data.total, 2)
        self.assertEqual(calls["bedesten"], 1)

    async def test_identical_searches_are_served_from_redis_cache_after_first_hit(self):
        calls = {"bedesten": 0}

        def bedesten_handler(request: httpx.Request) -> httpx.Response:
            calls["bedesten"] += 1
            return httpx.Response(200, json=sample_search_response())

        redis = FakeUpstashRedis()
        with patched_env(
            BEDESTEN_RATE_BACKEND="redis",
            BEDESTEN_CACHE_BACKEND="redis",
            UPSTASH_REDIS_REST_URL="https://redis.test",
            UPSTASH_REDIS_REST_TOKEN="test-token",
        ):
            client = BedestenApiClient()
            await attach_transports(
                client,
                httpx.MockTransport(bedesten_handler),
                redis.transport(),
            )
            try:
                first = await client.search_documents(sample_search_request())
                second = await client.search_documents(sample_search_request())
            finally:
                await client.close_client_session()

        self.assertEqual(first.data.total, 1)
        self.assertEqual(second.data.total, 1)
        self.assertEqual(calls["bedesten"], 1)

    async def test_bedesten_429_sets_global_redis_pause_before_next_request(self):
        calls = {"bedesten": 0}

        def bedesten_handler(request: httpx.Request) -> httpx.Response:
            calls["bedesten"] += 1
            return httpx.Response(429, headers={"Retry-After": "30"}, json={})

        redis = FakeUpstashRedis()
        with patched_env(
            BEDESTEN_RATE_BACKEND="redis",
            BEDESTEN_CACHE_BACKEND="off",
            BEDESTEN_RATE_MAX_WAIT_S="0.05",
            UPSTASH_REDIS_REST_URL="https://redis.test",
            UPSTASH_REDIS_REST_TOKEN="test-token",
        ):
            client = BedestenApiClient()
            await attach_transports(
                client,
                httpx.MockTransport(bedesten_handler),
                redis.transport(),
            )
            try:
                with self.assertRaises(httpx.HTTPStatusError):
                    await client.search_documents(sample_search_request())

                with self.assertRaises(BedestenRateLimited):
                    await client.search_documents(sample_search_request())
            finally:
                await client.close_client_session()

        self.assertEqual(calls["bedesten"], 1)

    async def test_redis_outage_does_not_block_bedesten_search(self):
        calls = {"bedesten": 0}

        def bedesten_handler(request: httpx.Request) -> httpx.Response:
            calls["bedesten"] += 1
            return httpx.Response(200, json=sample_search_response(total=3))

        redis = FakeUpstashRedis(fail=True)
        with patched_env(
            BEDESTEN_RATE_BACKEND="redis",
            BEDESTEN_CACHE_BACKEND="redis",
            UPSTASH_REDIS_REST_URL="https://redis.test",
            UPSTASH_REDIS_REST_TOKEN="test-token",
        ):
            client = BedestenApiClient()
            await attach_transports(
                client,
                httpx.MockTransport(bedesten_handler),
                redis.transport(),
            )
            try:
                response = await client.search_documents(sample_search_request())
            finally:
                await client.close_client_session()

        self.assertEqual(response.data.total, 3)
        self.assertEqual(calls["bedesten"], 1)


if __name__ == "__main__":
    unittest.main()
