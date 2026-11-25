"""The read-only key must not be able to write.

It is derived from the full key and handed to read-only clients, and the auth
middleware lets it through to /mcp so it can call the read tools - so the
write gate inside those tools is the only thing standing between it and the
data.
"""
import json

import pytest

from conftest import FULL_KEY, READ_KEY, call_tool, tool_json, tool_text

READ_TOOLS = [
    ("list_documents", {}),
    ("get_document_status", {"doc_id": "missing"}),
    ("search_pages", {"query": "anything"}),
]

WRITE_TOOLS = [
    ("upload_document", {"content_base64": "aGk=", "filename": "sneaky.txt"}),
    ("create_project", {"name": "sneaky"}),
    ("rename_project", {"name": "sneaky", "new_name": "sneakier"}),
    ("move_document", {"doc_id": "missing", "project": "sneaky"}),
    ("rename_document", {"doc_id": "missing", "name": "sneaky"}),
    ("retry_document", {"doc_id": "missing"}),
    ("extract_figures", {"doc_id": "missing"}),
    ("delete_document", {"doc_id": "missing"}),
    ("delete_project", {"name": "sneaky"}),
]


@pytest.mark.parametrize("name,arguments", READ_TOOLS)
def test_read_only_key_may_read(client, name, arguments):
    assert not call_tool(client, READ_KEY, name, arguments).get("isError")


@pytest.mark.parametrize("name,arguments", WRITE_TOOLS)
def test_read_only_key_cannot_write(client, name, arguments):
    result = call_tool(client, READ_KEY, name, arguments)
    assert result["isError"], result
    assert "full API key" in tool_text(result)
    assert "sneaky" not in json.dumps(server_db())


@pytest.mark.parametrize("name,arguments", WRITE_TOOLS)
def test_full_key_passes_the_write_gate(client, name, arguments):
    # Most of these hit a missing document and answer with an error payload -
    # what matters is that none of them is refused by the gate.
    assert not call_tool(client, FULL_KEY, name, arguments).get("isError")


def test_unknown_key_is_rejected(client):
    response = client.post("/api/state", headers={"Authorization": "Bearer nonsense"})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/projects"),
        ("PATCH", "/api/projects/x"),
        ("DELETE", "/api/projects/x"),
        ("POST", "/api/upload"),
        ("PATCH", "/api/documents/x"),
        ("POST", "/api/documents/x/retry"),
        ("POST", "/api/documents/x/media"),
        ("POST", "/api/media/backfill"),
        ("DELETE", "/api/documents/x"),
    ],
)
def test_read_only_key_cannot_write_over_rest(client, method, path):
    response = client.request(method, path, headers={"Authorization": f"Bearer {READ_KEY}"})
    assert response.status_code == 401


def test_read_only_key_may_read_over_rest(client):
    response = client.get("/api/state", headers={"Authorization": f"Bearer {READ_KEY}"})
    assert response.status_code == 200
    assert "documents" in response.json()


def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


def server_db():
    import server

    return server.store.load_db()


def test_tools_list_exposes_the_write_tools(client):
    response = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {FULL_KEY}", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    names = {t["name"] for t in response.json()["result"]["tools"]}
    assert {name for name, _ in WRITE_TOOLS} | {"get_document_status"} <= names


def test_gate_message_does_not_leak_a_key(client):
    text = tool_text(call_tool(client, READ_KEY, "delete_document", {"doc_id": "missing"}))
    assert FULL_KEY not in text and READ_KEY not in text


def test_duplicate_upload_is_not_a_write_for_the_read_key(client):
    # The read key must not even get as far as the deduplication lookup.
    result = call_tool(client, READ_KEY, "upload_document", {"content_base64": "aGk=", "filename": "probe.txt"})
    assert result["isError"]
    assert "doc_id" not in tool_text(result)
    assert tool_json(call_tool(client, FULL_KEY, "list_documents")) is not None
