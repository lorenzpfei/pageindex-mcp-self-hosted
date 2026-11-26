import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

os.environ["PAGEINDEX_DATA_DIR"] = tempfile.mkdtemp(prefix="pageindex-tests-")
os.environ["PAGEINDEX_MCP_API_KEY"] = "test-full-key"
os.environ["PAGEINDEX_FIGURE_MODEL"] = "off"

import pytest  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import server  # noqa: E402

FULL_KEY = "test-full-key"
READ_KEY = server.READ_API_KEY


@pytest.fixture(scope="session")
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def call_tool(client, key, name, arguments=None):
    response = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": arguments or {}}},
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def tool_text(result):
    return result["content"][0]["text"]


def tool_json(result):
    return json.loads(tool_text(result))


def upload(client, key, content: bytes, filename: str, project: str = ""):
    import base64

    return tool_json(call_tool(client, key, "upload_document", {
        "content_base64": base64.b64encode(content).decode(),
        "filename": filename,
        "project": project,
    }))
