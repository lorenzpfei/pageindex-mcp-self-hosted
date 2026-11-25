"""Round trip through the write tools, plus the two limits that protect the
box: deduplication by content hash and the upload size cap."""
import base64

import pytest

import documents
import server
from conftest import FULL_KEY, call_tool, tool_json, upload


def test_upload_move_rename_and_delete(client):
    created = upload(client, FULL_KEY, b"first line\nsecond line\n", "notes.txt", project="Round Trip")
    doc_id = created["doc_id"]
    assert created["status"] == "done"

    status = tool_json(call_tool(client, FULL_KEY, "get_document_status", {"doc_id": doc_id}))
    assert status["status"] == "done"
    assert status["type"] == "text"

    pages = tool_json(call_tool(client, FULL_KEY, "get_page_content", {"doc_id": doc_id, "pages": "2"}))
    assert pages[0]["content"] == "second line"

    tool_json(call_tool(client, FULL_KEY, "rename_document", {"doc_id": doc_id, "name": "Renamed notes"}))
    tool_json(call_tool(client, FULL_KEY, "move_document", {"doc_id": doc_id, "project": "Archive"}))
    entry = server.store.load_registry()[doc_id]
    assert entry["doc_name"] == "Renamed notes"
    assert entry["project"] == "Archive"

    tool_json(call_tool(client, FULL_KEY, "delete_document", {"doc_id": doc_id}))
    assert doc_id not in server.store.load_registry()


def test_identical_bytes_in_one_folder_reuse_the_document(client):
    content = b"dedup me\n"
    first = upload(client, FULL_KEY, content, "dedup.txt", project="Dedup")
    again = upload(client, FULL_KEY, content, "dedup-copy.txt", project="Dedup")
    assert again["doc_id"] == first["doc_id"]
    assert again["duplicate"] is True

    elsewhere = upload(client, FULL_KEY, content, "dedup.txt", project="Dedup Other")
    assert elsewhere["doc_id"] != first["doc_id"]
    assert "duplicate" not in elsewhere


def test_documents_without_a_hash_do_not_break_deduplication(client):
    created = upload(client, FULL_KEY, b"legacy\n", "legacy.txt", project="Legacy")
    server.store.update_document(created["doc_id"], sha256=None)
    again = upload(client, FULL_KEY, b"legacy\n", "legacy.txt", project="Legacy")
    assert again["doc_id"] != created["doc_id"]


def test_upload_over_the_size_cap_is_refused(client, monkeypatch):
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 16)
    payload = upload(client, FULL_KEY, b"x" * 64, "too-big.txt", project="Limits")
    assert "too large" in payload["error"]

    response = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {FULL_KEY}"},
        files={"file": ("too-big.txt", b"x" * 64, "text/plain")},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["error"]


def test_project_tools(client):
    tool_json(call_tool(client, FULL_KEY, "create_project", {"name": "Folder A"}))
    assert "Folder A" in server.store.load_db()["projects"]

    tool_json(call_tool(client, FULL_KEY, "rename_project", {"name": "Folder A", "new_name": "Folder B"}))
    projects = server.store.load_db()["projects"]
    assert "Folder B" in projects and "Folder A" not in projects

    tool_json(call_tool(client, FULL_KEY, "delete_project", {"name": "Folder B"}))
    assert "Folder B" not in server.store.load_db()["projects"]


def test_a_folder_holding_documents_is_not_deleted(client):
    created = upload(client, FULL_KEY, b"keep me\n", "keep.txt", project="Occupied")
    refused = tool_json(call_tool(client, FULL_KEY, "delete_project", {"name": "Occupied"}))
    assert "not empty" in refused["error"]
    assert "Occupied" in server.store.load_db()["projects"]
    tool_json(call_tool(client, FULL_KEY, "delete_document", {"doc_id": created["doc_id"]}))


def test_retry_rejects_text_documents(client):
    created = upload(client, FULL_KEY, b"text\n", "retry.txt", project="Retry")
    refused = tool_json(call_tool(client, FULL_KEY, "retry_document", {"doc_id": created["doc_id"]}))
    assert refused["error"] == "text documents are not ingested"


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("get_document_status", {"doc_id": "missing"}),
        ("rename_document", {"doc_id": "missing", "name": "x"}),
        ("move_document", {"doc_id": "missing", "project": "x"}),
        ("retry_document", {"doc_id": "missing"}),
        ("delete_document", {"doc_id": "missing"}),
    ],
)
def test_missing_documents_answer_with_an_error_payload(client, name, arguments):
    assert tool_json(call_tool(client, FULL_KEY, name, arguments))["error"]


def test_base64_must_be_valid(client):
    broken = tool_json(call_tool(client, FULL_KEY, "upload_document", {
        "content_base64": "not base64 at all!", "filename": "x.txt",
    }))
    assert "not valid base64" in broken["error"]


def test_exactly_one_source_is_required(client):
    both = tool_json(call_tool(client, FULL_KEY, "upload_document", {
        "url": "https://example.com/x.txt",
        "content_base64": base64.b64encode(b"x").decode(),
        "filename": "x.txt",
    }))
    assert "either url or content_base64" in both["error"]
    neither = tool_json(call_tool(client, FULL_KEY, "upload_document", {}))
    assert "either url or content_base64" in neither["error"]
