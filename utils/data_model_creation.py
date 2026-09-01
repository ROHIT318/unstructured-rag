"""
Data model markdown database file — specs/create_db_markdown_file_txt.md and
specs/create_db_markdown_file_rest.md.

One markdown file per data model under data/data_model_description/ records the
model's details (name, description, created / last modified) and one entry per
ingested file. txt, image, table, pdf and ppt entries are fully detailed; every
other type gets a minimal placeholder entry. All functions are headless — UI
messages stay in the upload form.
"""

import hashlib
import os
import re
from datetime import datetime

import chromadb

from utils.data_pipeline import sanitize_collection_name

DESCRIPTION_FOLDER = "data/data_model_description/"
FUTURE_ITERATION_NOTE = "Detailed tracking for this file type is planned for a future iteration."


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def description_filename(model_name: str) -> str:
    sanitized = sanitize_collection_name(model_name)
    if len(sanitized) <= 100:
        return f"{sanitized}.md"
    digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized[:87]}_{digest}.md"


def description_path(model_name: str) -> str:
    return f"{DESCRIPTION_FOLDER}{description_filename(model_name)}"


def _read_description(model_name: str) -> str | None:
    path = description_path(model_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _txt_entry(record: dict, model_name: str) -> str:
    collection_name = record.get("collection_name") or sanitize_collection_name(model_name)
    ids = ", ".join(record.get("document_ids") or [])
    return (
        f"### {record['file_name']} — txt\n"
        f"- **File type:** txt\n"
        f"- **Collection name:** {collection_name}\n"
        f"- **File name:** {record['file_name']}\n"
        f"- **Document ids:** {ids}\n"
        f"- **Description:**\n"
    )


def _image_entry(record: dict, model_name: str) -> str:
    collection_name = record.get("collection_name") or sanitize_collection_name(f"{model_name}_images")
    ids = ", ".join(record.get("document_ids") or [])
    return (
        f"### {record['file_name']} — image\n"
        f"- **File type:** image\n"
        f"- **Collection name:** {collection_name}\n"
        f"- **File name:** {record['file_name']}\n"
        f"- **File path:** {record.get('file_path', '')}\n"
        f"- **Document ids:** {ids}\n"
        f"- **Description:**\n"
    )


def _document_entry(record: dict, model_name: str, source_type: str) -> str:
    """Shared pdf/ppt entry — ids of the text chunks plus the extracted images, when present."""
    collection_name = record.get("collection_name") or sanitize_collection_name(model_name)
    ids = ", ".join(record.get("document_ids") or [])
    lines = [
        f"### {record['file_name']} — {source_type}",
        f"- **File type:** {source_type}",
        f"- **Collection name:** {collection_name}",
        f"- **File name:** {record['file_name']}",
        f"- **Document ids:** {ids}",
    ]
    images = record.get("images") or []
    if images:
        lines.append("- **Images:**")
        lines.extend(f"  - `{image['file_path']}` (id: {image['document_id']})" for image in images)
    lines.append("- **Description:**")
    return "\n".join(lines) + "\n"


def _rows_markdown(rows: list, columns: list, indent: str = "    ") -> str:
    """Top rows as an indented markdown table; pipes and newlines inside cells are escaped."""
    if not rows or not columns:
        return f"{indent}*(no rows)*"

    def cell(value) -> str:
        return "" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(indent + line for line in lines)


def _table_entry(record: dict, model_name: str) -> str:
    collection_name = record.get("collection_name") or sanitize_collection_name(model_name)
    lines = [
        f"### {record['file_name']} — table",
        f"- **File type:** table",
        f"- **Collection name:** {collection_name}",
        f"- **File name:** {record['file_name']}",
        f"- **Tables:**",
    ]
    for table in record.get("tables") or []:
        columns = [str(column) for column in table.get("column_names") or []]
        lines.append("")
        lines.append(f"  #### {table.get('table_name', '')}")
        lines.append(f"  - **Csv file path:** {table.get('csv_path', '')}")
        lines.append("  - **Top 5 rows:**")
        lines.append("")
        lines.append(_rows_markdown(table.get("top_5_rows") or [], columns))
        lines.append("")
        lines.append(f"  - **Column names:** {', '.join(columns)}")
        lines.append(f"  - **Column data types:** {', '.join(str(entry) for entry in table.get('column_types') or [])}")
        lines.append("  - **Description:**")
    return "\n".join(lines) + "\n"


def _entry_for(record: dict, model_name: str) -> str:
    source_type = record.get("source_type")
    if source_type == "txt":
        return _txt_entry(record, model_name)
    if source_type == "image":
        return _image_entry(record, model_name)
    if source_type == "table":
        return _table_entry(record, model_name)
    if source_type in ("pdf", "ppt"):
        return _document_entry(record, model_name, source_type)
    return _placeholder_entry(record)


def _placeholder_entry(record: dict) -> str:
    source_type = record.get("source_type", "skipped")
    return (
        f"### {record['file_name']} — {source_type}\n"
        f"- **File type:** {source_type}\n"
        f"- {FUTURE_ITERATION_NOTE}\n"
    )


def data_model_creation(model_name: str, description: str, ingestion_records: list = None) -> str:
    """
    Create or update the model's markdown database file after a submission.

    First submission creates the file; later submissions append entries,
    replace the model description (an empty new value never erases a recorded
    one), keep Created, and refresh Last modified. Existing entries are never
    rewritten (§6). Returns the file path.
    """
    records = ingestion_records or []
    path = description_path(model_name)
    os.makedirs(DESCRIPTION_FOLDER, exist_ok=True)
    now = _now()

    content = _read_description(model_name)
    if not content or "# Model:" not in content:
        # Missing or corrupt file — deterministically rewrite it (§6).
        content = (
            f"# Model: {model_name}\n\n"
            f"**Description:** {description.strip() or '*(none provided)*'}\n\n"
            f"**Created:** {now}\n"
            f"**Last modified:** {now}\n\n"
            f"## Files\n\n"
        )
    else:
        if description.strip():
            content = re.sub(r"\*\*Description:\*\*.*", f"**Description:** {description.strip()}", content, count=1)
        content = re.sub(r"\*\*Last modified:\*\*.*", f"**Last modified:** {now}", content, count=1)
        if "## Files" not in content:
            content = content.rstrip("\n") + "\n\n## Files\n\n"
        elif not content.endswith("\n\n"):
            content = content.rstrip("\n") + "\n\n"

    entries = [_entry_for(record, model_name) for record in records]
    content = content + "".join(f"{entry}\n" for entry in entries)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _entry_blocks(content: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of every `### <file name> — <type>` block under ## Files."""
    blocks = []
    starts = [match.start() for match in re.finditer(r"^### .*$", content, flags=re.MULTILINE)]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(content)
        blocks.append((start, end, content[start:end]))
    return blocks


def _entry_type(block: str) -> str | None:
    """The type marker of an entry block — the word after the last ` — ` of its header line."""
    header = block.split("\n", 1)[0]
    match = re.match(r"^### .* — (\w+)\s*$", header)
    return match.group(1) if match else None


def _ids_from(block: str) -> list:
    ids_line = re.search(r"\*\*Document ids:\*\* (.*)", block)
    if not ids_line:
        return []
    return [document_id.strip() for document_id in ids_line.group(1).split(",") if document_id.strip()]


def remove_file_entry(model_name: str, file_name: str, vector_db_path: str = "data/vector_db/") -> dict:
    """
    Remove one file's entry from the model's markdown file (§6.2).

    The entry's Chroma documents are deleted in the same operation — cross-checked
    against the file-name metadata so documents of a re-uploaded same-named file
    survive — and so are the file's permanent disk artifacts (image files under
    data/images/, table csvs under data/tables/), which unlike txt temp files are
    kept on purpose. Only the most recent entry for the file name is removed,
    since re-uploads create separate entries (§4). Returns a summary dict.
    """
    summary = {"entry_removed": False, "documents_deleted": 0, "files_removed": 0}
    content = _read_description(model_name)
    if content is None:
        return summary

    matching = [block for block in _entry_blocks(content) if block[2].startswith(f"### {file_name} —")]
    if not matching:
        return summary
    start, end, block = matching[-1]

    entry_type = _entry_type(block)
    collection_match = re.search(r"\*\*Collection name:\*\* (\S+)", block)

    if entry_type == "image":
        if collection_match:
            summary["documents_deleted"] += _delete_documents(collection_match.group(1), _ids_from(block), file_name, vector_db_path)
        file_path = re.search(r"\*\*File path:\*\* (\S+)", block)
        summary["files_removed"] += _remove_files([file_path.group(1)] if file_path else [])

    elif entry_type in ("pdf", "ppt"):
        if collection_match:
            summary["documents_deleted"] += _delete_documents(collection_match.group(1), _ids_from(block), file_name, vector_db_path)
        # The entry's images live in the model's image collection and on disk.
        image_collection = sanitize_collection_name(f"{model_name}_images")
        image_ids = re.findall(r"\(id: (\S+)\)", block)
        image_paths = re.findall(r"- `([^`]+)` \(id:", block)
        if image_ids:
            summary["documents_deleted"] += _delete_documents(image_collection, image_ids, file_name, vector_db_path)
        summary["files_removed"] += _remove_files(image_paths)

    elif entry_type == "table":
        # Table entries record no ids; their documents are found via the file-name metadata.
        if collection_match:
            summary["documents_deleted"] += _delete_documents_by_metadata(collection_match.group(1), file_name, vector_db_path)
        csv_paths = re.findall(r"\*\*Csv file path:\*\* (\S+)", block)
        summary["files_removed"] += _remove_files(csv_paths)

    elif entry_type == "txt" and collection_match:
        summary["documents_deleted"] += _delete_documents(collection_match.group(1), _ids_from(block), file_name, vector_db_path)

    now = _now()
    content = (content[:start] + content[end:]).rstrip("\n") + "\n"
    content = re.sub(r"\*\*Last modified:\*\*.*", f"**Last modified:** {now}", content, count=1)
    with open(description_path(model_name), "w", encoding="utf-8") as f:
        f.write(content)

    summary["entry_removed"] = True
    return summary


def _remove_files(paths: list) -> int:
    """Delete the given disk files, skipping anything missing or undeletable."""
    removed = 0
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed


def _delete_documents_by_metadata(collection_name: str, file_name: str, vector_db_path: str) -> int:
    """Delete every document whose metadata names the given file (used where no ids are recorded)."""
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        collection = client.get_collection(collection_name)
        existing = collection.get(where={"file_name": file_name})
    except Exception:
        return 0

    ids = existing.get("ids") or []
    if not ids:
        return 0
    try:
        collection.delete(ids=ids)
    except Exception:
        return 0
    return len(ids)


def _delete_documents(collection_name: str, ids: list, file_name: str, vector_db_path: str) -> int:
    """Delete the listed ids, keeping those whose file-name metadata points at another file."""
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        collection = client.get_collection(collection_name)
        existing = collection.get(ids=ids, include=["metadatas"])
    except Exception:
        return 0

    deletable = [
        document_id
        for document_id, metadata in zip(existing["ids"], existing.get("metadatas") or [])
        if not metadata or metadata.get("file_name") == file_name
    ]
    if not deletable:
        return 0
    try:
        collection.delete(ids=deletable)
    except Exception:
        return 0
    return len(deletable)


def delete_model_record(model_name: str) -> bool:
    """Delete the model's markdown file (whole-model deletion, §6.2). Collections are removed elsewhere."""
    path = description_path(model_name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _touch(content: str) -> str:
    """Refresh the header's Last modified line."""
    return re.sub(r"\*\*Last modified:\*\*.*", f"**Last modified:** {_now()}", content, count=1)


def _write_description(model_name: str, content: str) -> None:
    with open(description_path(model_name), "w", encoding="utf-8") as f:
        f.write(content)


def read_model_description(model_name: str) -> str | None:
    """The model's markdown record, or None when no file exists (specs/data_model_editing.md)."""
    return _read_description(model_name)


def model_display_name(model_name: str) -> str | None:
    """The display name recorded in the model's markdown header (`# Model: <name>`), or None."""
    content = _read_description(model_name)
    if content is None:
        return None
    match = re.search(r"^# Model: (.+)$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def recorded_models() -> list[str]:
    """
    Underlying names (markdown file stems) of every model that has a record
    file — including models with no collections at all (e.g. created with no
    files uploaded). The sidebar unions these with the collection-derived
    models so such models stay reachable and deletable.
    """
    if not os.path.isdir(DESCRIPTION_FOLDER):
        return []
    return sorted(filename[: -len(".md")] for filename in os.listdir(DESCRIPTION_FOLDER)
                  if filename.endswith(".md"))


def _recorded_display_names() -> dict[str, str]:
    """{display name: markdown file stem} of every recorded model — the base for uniqueness checks."""
    names = {}
    if not os.path.isdir(DESCRIPTION_FOLDER):
        return names
    for filename in os.listdir(DESCRIPTION_FOLDER):
        if not filename.endswith(".md"):
            continue
        try:
            with open(f"{DESCRIPTION_FOLDER}{filename}", "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        match = re.search(r"^# Model: (.+)$", content, flags=re.MULTILINE)
        if match:
            names[match.group(1).strip()] = filename[: -len(".md")]
    return names


def update_model_display_name(model_name: str, new_name: str) -> str:
    """
    Change the display name recorded in the model's markdown header (§2 of
    specs/data_model_editing.md — display-name only: the collection names and
    the markdown file name stay untouched). Raises ValueError when the new name
    is empty, already used by another model, or the model has no record file.
    """
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("The model name cannot be empty.")

    taken = _recorded_display_names()
    own_stem = description_filename(model_name)[: -len(".md")]
    if new_name in taken and taken[new_name] != own_stem:
        raise ValueError(f"Another model is already named \"{new_name}\".")

    content = _read_description(model_name)
    if content is None:
        raise ValueError("This model has no record file to rename.")

    content = re.sub(r"^# Model: .+$", f"# Model: {new_name}", content, count=1, flags=re.MULTILINE)
    _write_description(model_name, _touch(content))
    return new_name


def update_model_description(model_name: str, new_description: str) -> bool:
    """Rewrite the header's Description line (§2). An empty value is a no-op returning False."""
    new_description = new_description.strip()
    content = _read_description(model_name)
    if content is None or not new_description:
        return False

    content = re.sub(r"\*\*Description:\*\*.*", f"**Description:** {new_description}", content, count=1)
    _write_description(model_name, _touch(content))
    return True


def _table_descriptions(block: str) -> list[dict]:
    """Per-table records of a table entry — one per `  #### <table name>` sub-block."""
    headers = list(re.finditer(r"^  #### (.+)$", block, flags=re.MULTILINE))
    tables = []
    for index, header in enumerate(headers):
        segment_end = headers[index + 1].start() if index + 1 < len(headers) else len(block)
        segment = block[header.start():segment_end]
        descriptions = re.findall(r"\*\*Description:\*\* ?(.*)", segment)
        tables.append({
            "table_name": header.group(1).strip(),
            "description": descriptions[-1].strip() if descriptions else "",
            "text": segment,
        })
    return tables


def read_model_entries(model_name: str) -> list[dict]:
    """
    One record per `### <file name> — <type>` entry of the model's markdown file (§3).
    An entry's description is the value of its last `**Description:**` line. Table
    entries carry no file-level description — each of their tables (xlsx sheets /
    csv) has its own, listed in `tables` and edited via update_table_description.
    """
    content = _read_description(model_name)
    if content is None:
        return []

    entries = []
    for _, _, block in _entry_blocks(content):
        header_match = re.match(r"^### (.+) — (\w+)\s*$", block.split("\n", 1)[0])
        if not header_match:
            continue
        source_type = header_match.group(2)
        entry = {
            "file_name": header_match.group(1).strip(),
            "source_type": source_type,
            "description": "",
            "text": block,
        }
        if source_type == "table":
            entry["tables"] = _table_descriptions(block)
        else:
            descriptions = re.findall(r"\*\*Description:\*\* ?(.*)", block)
            entry["description"] = descriptions[-1].strip() if descriptions else ""
        entries.append(entry)
    return entries


def _replace_last_description(text: str, description: str) -> str | None:
    """
    Replace the last `**Description:**` line of the text with the new value,
    keeping the line's list prefix (`- ` / `  - `). An empty value clears it.
    Returns None when the text has no Description line.
    """
    matches = list(re.finditer(r"^(\s*- )?\*\*Description:\*\*.*$", text, flags=re.MULTILINE))
    if not matches:
        return None
    last = matches[-1]
    replacement = f"{last.group(1) or ''}**Description:** {description.strip()}".rstrip()
    return text[:last.start()] + replacement + text[last.end():]


def update_file_description(model_name: str, file_name: str, description: str) -> bool:
    """
    Set — or clear, when the value is empty — the description of the file's most
    recent entry (§3). Table entries are refused: their tables are described
    individually via update_table_description. Returns False when the file has
    no entry, no record file, or is a table entry.
    """
    content = _read_description(model_name)
    if content is None:
        return False

    matching = [block for block in _entry_blocks(content) if block[2].startswith(f"### {file_name} —")]
    if not matching:
        return False
    start, end, block = matching[-1]

    if re.search(r"^  #### ", block, flags=re.MULTILINE):
        return False

    new_block = _replace_last_description(block, description)
    if new_block is None:
        # Entries without a Description line (e.g. skipped files, empty table
        # files) get one appended.
        new_block = block.rstrip("\n") + f"\n- **Description:** {description.strip()}".rstrip() + "\n"
    content = content[:start] + new_block + content[end:]
    _write_description(model_name, _touch(content))
    return True


def update_table_description(model_name: str, file_name: str, table_name: str, description: str) -> bool:
    """
    Set — or clear, when the value is empty — the description of one table of a
    table entry (§3): locates the file's most recent entry, the table's
    `#### <table name>` sub-block inside it, and replaces that sub-block's
    Description line. Returns False when the file, entry, or table cannot be
    found.
    """
    content = _read_description(model_name)
    if content is None:
        return False

    matching = [block for block in _entry_blocks(content) if block[2].startswith(f"### {file_name} —")]
    if not matching:
        return False
    start, end, block = matching[-1]

    headers = list(re.finditer(r"^  #### (.+)$", block, flags=re.MULTILINE))
    target = next((header for header in reversed(headers) if header.group(1).strip() == table_name), None)
    if target is None:
        return False
    segment_end = next((header.start() for header in headers if header.start() > target.start()), len(block))

    new_segment = _replace_last_description(block[target.start():segment_end], description)
    if new_segment is None:
        return False
    content = content[:start] + block[:target.start()] + new_segment + block[segment_end:] + content[end:]
    _write_description(model_name, _touch(content))
    return True


def delete_model(model_name: str, vector_db_path: str = "data/vector_db/") -> dict:
    """
    Delete the whole model (§6 of specs/data_model_editing.md): both Chroma
    collections, the disk files they reference (only under data/images/ and
    data/tables/), and the markdown record. Each step is attempted
    independently — a missing collection or file is not an error. Returns a
    summary dict of what was removed, with an "errors" list naming anything
    that existed but could not be deleted (so failures surface in the UI
    instead of being silently skipped).
    """
    summary = {"collections_deleted": [], "files_removed": 0, "description_removed": False, "errors": []}
    sanitized = sanitize_collection_name(model_name)
    client = chromadb.PersistentClient(path=vector_db_path)

    # Disk files to remove: every data/images/ and data/tables/ path mentioned
    # in the model's record, plus the ones the collections' documents point at.
    # Documents are read before their collection is deleted; the record paths
    # are the fallback that keeps file cleanup working even when a collection
    # cannot be read.
    record = _read_description(model_name) or ""
    referenced_paths = re.findall(r"data/(?:images|tables)/\S+", record)

    existing_collections = {collection.name for collection in client.list_collections()}
    for collection_name in (sanitized, f"{sanitized}_images"):
        if collection_name not in existing_collections:
            continue
        try:
            collection = client.get_collection(collection_name)
            documents = (collection.get(include=["documents"]) or {}).get("documents") or []
            for document in documents:
                referenced_paths.extend(_paths_referenced_by(document))
        except Exception as error:
            summary["errors"].append(f"could not read collection {collection_name}: {error}")
            continue
        try:
            client.delete_collection(collection_name)
            summary["collections_deleted"].append(collection_name)
        except Exception as error:
            summary["errors"].append(f"could not delete collection {collection_name}: {error}")

    summary["files_removed"] = _remove_files(referenced_paths)
    summary["description_removed"] = delete_model_record(model_name)
    return summary


def _paths_referenced_by(document) -> list[str]:
    """
    Disk files a stored document points at: an image document is the file path
    itself, and a table pointer document carries a `Path:` line. Anything else
    (prose chunks) references nothing — and only data/images/ / data/tables/
    paths are ever eligible.
    """
    if not isinstance(document, str):
        return []
    if document.startswith(("data/images/", "data/tables/")):
        return [document]
    match = re.search(r"^Path: (\S+)$", document, flags=re.MULTILINE)
    path = match.group(1) if match else ""
    return [path] if path.startswith("data/tables/") else []
