import importlib
import unittest

from starlette.testclient import TestClient


class HealthRouteTests(unittest.TestCase):
    def test_simple_asgi_app_exposes_health_routes(self):
        module = importlib.import_module("app")
        client = TestClient(module.app)

        for path in ("/health", "/health/"):
            response = client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")

    def test_fastapi_wrapper_exposes_health_routes(self):
        module = importlib.import_module("asgi_app")
        client = TestClient(module.app)

        for path in ("/health", "/health/"):
            response = client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
