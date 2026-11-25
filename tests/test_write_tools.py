"""The read-only key must not be able to write.

It is derived from the full key and handed to read-only clients, and the
auth middleware lets it through to /mcp so it can call the read tools - so
the write tools are the only thing standing between it and the data.
"""
import base64
import json

import pytest
from starlette.testclient import TestClient

import server

FULL_KEY = "test-full-key"


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def call_tool(client, key, name, arguments):
    response = client.post(
        "/mcp",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json, text/event-stream",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def text_of(result):
    return result["content"][0]["text"]


def test_read_only_key_may_read(client):
    result = call_tool(client, server.READ_API_KEY, "list_documents", {})
    assert not result.get("isError")
    assert isinstance(json.loads(text_of(result)), list)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("upload_document", {"content_base64": "aGk=", "filename": "hi.txt"}),
        ("create_project", {"name": "sneaky"}),
        ("move_document", {"doc_id": "whatever", "project": "sneaky"}),
        ("delete_document", {"doc_id": "whatever"}),
    ],
)
def test_read_only_key_cannot_write(client, name, arguments):
    result = call_tool(client, server.READ_API_KEY, name, arguments)
    assert result["isError"], result
    assert "full API key" in text_of(result)
    assert "sneaky" not in json.dumps(server.store.load_db())


def test_full_key_uploads_base64_document(client):
    content = "first line\nsecond line\n"
    result = call_tool(
        client,
        FULL_KEY,
        "upload_document",
        {
            "content_base64": base64.b64encode(content.encode()).decode(),
            "filename": "notes.txt",
            "project": "Uploads",
        },
    )
    doc_id = json.loads(text_of(result))["doc_id"]

    listed = {d["doc_id"]: d for d in json.loads(text_of(call_tool(client, FULL_KEY, "list_documents", {})))}
    assert listed[doc_id]["status"] == "done"
    assert listed[doc_id]["project"] == "Uploads"

    pages = json.loads(text_of(call_tool(client, FULL_KEY, "get_page_content", {"doc_id": doc_id, "pages": "2"})))
    assert pages[0]["content"] == "second line"

    call_tool(client, FULL_KEY, "move_document", {"doc_id": doc_id, "project": "Archive"})
    assert server.store.load_registry()[doc_id]["project"] == "Archive"

    call_tool(client, FULL_KEY, "delete_document", {"doc_id": doc_id})
    assert doc_id not in server.store.load_registry()


def test_multipart_upload_shares_the_tool_path(client):
    response = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {FULL_KEY}", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert "upload_document" in [t["name"] for t in response.json()["result"]["tools"]]

    uploaded = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {FULL_KEY}"},
        files={"file": ("via-form.txt", b"hello\n", "text/plain")},
        data={"project": "Uploads"},
    )
    assert uploaded.status_code == 201, uploaded.text
    entry = server.store.load_registry()[uploaded.json()["doc_id"]]
    assert entry["doc_name"] == "via-form.txt"
    assert entry["project"] == "Uploads"


def test_read_only_key_cannot_use_the_upload_route(client):
    refused = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {server.READ_API_KEY}"},
        files={"file": ("sneaky.txt", b"hello\n", "text/plain")},
    )
    assert refused.status_code == 401
