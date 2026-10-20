"""Tests for the MCP OAuth helpers."""

import json
import unittest
from unittest import mock

import httpx2 as httpx

from mistralai.extra.mcp.auth import build_oauth_params, get_oauth_server_metadata
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams

SERVER_URL = "https://mcp.example"
REDIRECT_URL = "http://localhost:8080/callback"


class TestFallbackServerMetadata(unittest.IsolatedAsyncioTestCase):
    """Servers without metadata discovery fall back to the default endpoints."""

    def _no_well_known_metadata(self):
        return mock.patch(
            "mistralai.extra.mcp.auth.get_well_known_authorization_server_metadata",
            mock.AsyncMock(return_value=None),
        )

    async def test_fallback_metadata_has_registration_endpoint(self) -> None:
        with self._no_well_known_metadata():
            metadata = await get_oauth_server_metadata(SERVER_URL)

        self.assertEqual(metadata.authorization_endpoint, f"{SERVER_URL}/authorize")
        self.assertEqual(metadata.token_endpoint, f"{SERVER_URL}/token")
        self.assertEqual(metadata.registration_endpoint, f"{SERVER_URL}/register")

    async def test_build_oauth_params_registers_client_on_fallback_endpoint(
        self,
    ) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                201, json={"client_id": "cid", "client_secret": "csecret"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as async_client:
            with self._no_well_known_metadata():
                params = await build_oauth_params(
                    SERVER_URL,
                    redirect_url=REDIRECT_URL,
                    async_client=async_client,
                )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(str(requests[0].url), f"{SERVER_URL}/register")
        self.assertEqual(
            json.loads(requests[0].content)["redirect_uris"], [REDIRECT_URL]
        )
        self.assertEqual(params.client_id, "cid")
        self.assertEqual(params.client_secret, "csecret")
        self.assertEqual(params.scheme.token_url, f"{SERVER_URL}/token")


class TestMCPClientSSEBaseUrl(unittest.TestCase):
    """base_url is the server URL used for OAuth discovery: the SSE URL without
    its trailing /sse path segment."""

    def _base_url(self, url: str) -> str:
        return MCPClientSSE(sse_params=SSEServerParams(url=url)).base_url

    def test_strips_sse_suffix(self) -> None:
        self.assertEqual(
            self._base_url("https://mcp.linear.app/sse"), "https://mcp.linear.app"
        )
        self.assertEqual(
            self._base_url("https://mcp.example/v1/sse/"), "https://mcp.example/v1"
        )

    def test_keeps_hosts_and_paths_ending_in_s_or_e(self) -> None:
        self.assertEqual(
            self._base_url("https://mcp.example.services/sse"),
            "https://mcp.example.services",
        )
        self.assertEqual(
            self._base_url("https://mcp.example/api/messages/sse"),
            "https://mcp.example/api/messages",
        )

    def test_url_without_sse_suffix_is_unchanged(self) -> None:
        self.assertEqual(
            self._base_url("https://mcp.example/events"), "https://mcp.example/events"
        )


if __name__ == "__main__":
    unittest.main()
