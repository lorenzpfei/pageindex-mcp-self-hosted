"""upload_document(url=...) makes the server fetch a URL of the caller's
choosing, so it must never be usable to reach the host, the docker network or
a cloud metadata endpoint - not on the first request and not through a
redirect."""
import asyncio
import ipaddress
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import documents
import server
from conftest import FULL_KEY, call_tool, tool_json

BLOCKED = [
    "127.0.0.1",          # loopback
    "0.0.0.0",            # unspecified
    "10.0.0.1",           # private
    "172.16.0.1",         # private
    "192.168.1.1",        # private
    "169.254.169.254",    # link-local: the cloud metadata endpoint
    "100.64.0.1",         # carrier-grade NAT
    "224.0.0.1",          # multicast
    "::1",                # IPv6 loopback
    "fc00::1",            # IPv6 unique-local
    "fe80::1",            # IPv6 link-local
    "ff02::1",            # IPv6 multicast
    "::ffff:127.0.0.1",   # IPv4-mapped loopback
    "::ffff:169.254.169.254",
]

PUBLIC = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"]


@pytest.mark.parametrize("address", BLOCKED)
def test_non_public_addresses_are_internal(address):
    assert server._is_internal(ipaddress.ip_address(address))


@pytest.mark.parametrize("address", PUBLIC)
def test_public_addresses_pass(address):
    assert not server._is_internal(ipaddress.ip_address(address))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/x",
        "http://127.0.0.1:8000/api/state",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x",
        "http://10.1.2.3/secrets",
        "https://192.168.0.5/admin",
    ],
)
def test_check_url_refuses(url):
    with pytest.raises(ValueError):
        asyncio.run(server._check_url(url))


def test_check_url_needs_a_host():
    with pytest.raises(ValueError):
        asyncio.run(server._check_url("http:///nohost"))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect-to-metadata":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
        elif self.path == "/redirect-loop":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop")
            self.end_headers()
        elif self.path == "/missing":
            self.send_response(404)
            self.end_headers()
        else:
            body = b"alpha\nbeta\n" * 8
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def origin():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def allow_loopback(monkeypatch):
    """The test origin lives on 127.0.0.1, which the guard blocks for good
    reason. Loopback is waved through so the download path itself can be
    exercised - every other address stays blocked, which is what makes the
    redirect test meaningful."""
    real = server._is_internal
    monkeypatch.setattr(server, "_is_internal", lambda ip: False if ip.is_loopback else real(ip))


def test_downloads_and_names_the_document_after_the_url(client, origin, allow_loopback):
    created = tool_json(call_tool(client, FULL_KEY, "upload_document", {
        "url": f"{origin}/papers/readme.md", "project": "Downloads",
    }))
    entry = server.store.load_registry()[created["doc_id"]]
    assert entry["doc_name"] == "readme.md"
    assert entry["project"] == "Downloads"
    with open(entry["pdf_path"], encoding="utf-8") as f:
        assert f.read().startswith("alpha\nbeta\n")
    tool_json(call_tool(client, FULL_KEY, "delete_document", {"doc_id": created["doc_id"]}))


def test_a_redirect_into_the_metadata_endpoint_is_blocked(client, origin, allow_loopback):
    refused = tool_json(call_tool(client, FULL_KEY, "upload_document", {
        "url": f"{origin}/redirect-to-metadata",
    }))
    assert "not a public address" in refused["error"]
    assert not any(e.get("project") == "" and "meta-data" in e.get("doc_name", "")
                   for e in server.store.load_registry().values())


def test_redirect_loops_give_up(client, origin, allow_loopback):
    refused = tool_json(call_tool(client, FULL_KEY, "upload_document", {"url": f"{origin}/redirect-loop"}))
    assert f"more than {server.MAX_REDIRECTS} redirects" in refused["error"]


def test_http_errors_are_reported(client, origin, allow_loopback):
    refused = tool_json(call_tool(client, FULL_KEY, "upload_document", {"url": f"{origin}/missing"}))
    assert "download failed" in refused["error"]


def test_the_download_stops_at_the_size_cap(client, origin, allow_loopback, monkeypatch):
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 8)
    refused = tool_json(call_tool(client, FULL_KEY, "upload_document", {"url": f"{origin}/big.txt"}))
    assert "download exceeds" in refused["error"]
