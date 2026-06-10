"""Plain-text document support (code, notebooks, markdown, any UTF-8 file).

Text documents skip the LLM tree-building entirely: they are stored as plain
text at upload time and served through the same MCP tools as PDFs, with
1-indexed line numbers taking the role of page numbers (same convention as
upstream PageIndex's Markdown support).

Jupyter notebooks are converted to readable text on upload: markdown cells
as-is, code cells fenced, outputs dropped (they bloat token counts and are
reproducible from the code).
"""
import json

MAX_TEXT_BYTES = 10 * 1024 * 1024


def convert_upload(filename: str, raw: bytes) -> str:
    """Decode an uploaded file to the plain text we store. Raises ValueError
    if the file is too large or not UTF-8 text."""
    if len(raw) > MAX_TEXT_BYTES:
        raise ValueError(f"text file too large (max {MAX_TEXT_BYTES // (1024 * 1024)} MB)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("file is not UTF-8 text - only PDFs and plain-text files are supported")
    if "\x00" in text:
        raise ValueError("file contains binary data")
    if filename.lower().endswith(".ipynb"):
        text = _ipynb_to_text(text)
    return text.replace("\r\n", "\n")


def _ipynb_to_text(raw: str) -> str:
    try:
        nb = json.loads(raw)
        cells = nb["cells"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("file is not a valid Jupyter notebook")
    lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
    parts = []
    for cell in cells:
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```{lang}\n{src}\n```")
        else:
            parts.append(src)
    return "\n\n".join(parts)


def line_content(path: str, line_nums: list[int]) -> str:
    """Return requested 1-indexed lines as JSON, grouped into contiguous runs:
    [{"lines": "5-7", "content": "..."}]. Mirrors the per-page output shape of
    PDF retrieval."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    valid = [n for n in line_nums if 1 <= n <= len(lines)]
    runs = []
    for n in valid:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    result = [
        {
            "lines": f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]),
            "content": "\n".join(lines[n - 1] for n in run),
        }
        for run in runs
    ]
    return json.dumps(result, ensure_ascii=False)
