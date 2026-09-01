"""
Editable data model details view — specs/data_model_editing.md.

Rendered in the main area of the Data Upload page when a model is selected in
the left pane: the model's name (display name) and description are editable,
every ingested file has an editable description and a delete button, new files
can be added to the model, and the whole model can be deleted. All mutations go
through the headless functions of utils/data_model_creation.py — this module
only renders and dispatches. Destructive actions use a two-step confirmation
(§5, §6) and never run on a single click.

Widget keys carry the underlying model name (and, for per-file widgets, the
file name and its occurrence among the entries) so editor state never leaks
from one model or file to another.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_model_creation import (
    data_model_creation,
    delete_model,
    model_display_name,
    read_model_description,
    read_model_entries,
    remove_file_entry,
    update_file_description,
    update_model_description,
    update_model_display_name,
    update_table_description,
)
from utils.data_model_ingestion import ingest_uploaded_files
from utils.data_model_sidebar import VECTOR_DB_PATH, list_models, model_collection_counts

UPLOADER_TYPES = ["csv", "xlsx", "xlsm", "xlsb", "txt", "jpeg", "jpg", "png", "pdf", "pptx"]
DETAILS_CSS_PATH = Path(__file__).parent / "data_model_details.css"


def _inject_details_css() -> None:
    """Entry-card layout fixes — kept in data_model_details.css, not inline strings."""
    try:
        css = DETAILS_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_details_view(model_name: str) -> None:
    """The editable model details view for the selected model."""
    _inject_details_css()

    message = st.session_state.pop("details_message", None)
    if message:
        st.success(message)

    content = read_model_description(model_name)
    if content is None:
        # No record file yet (model created before records existed, or never
        # recorded): counts only, plus add-files — which creates the record.
        _render_no_record_view(model_name)
        return

    _render_recorded_view(model_name, content)


def _render_recorded_view(model_name: str, content: str) -> None:
    display_name = model_display_name(model_name) or model_name
    last_modified = re.search(r"\*\*Last modified:\*\* (.*)", content)

    st.header(display_name)
    if last_modified:
        st.caption(f"Last modified: {last_modified.group(1).strip()}")

    _render_name_editor(model_name, display_name)
    _render_description_editor(model_name, content)
    _render_file_entries(model_name)
    _render_add_files(model_name, display_name)
    _render_delete_model(model_name, display_name)


def _render_no_record_view(model_name: str) -> None:
    st.header(model_name)
    st.markdown("**Description:** *(no description recorded)*")

    counts = model_collection_counts(model_name)
    if "text" in counts:
        st.write(f"Text collection: {counts['text']} document(s)")
    if "images" in counts:
        st.write(f"Image collection: {counts['images']} document(s)")
    st.caption("This model has no record file — add files below to create one.")

    _render_add_files(model_name, model_name)
    _render_delete_model(model_name, model_name)


def _render_name_editor(model_name: str, display_name: str) -> None:
    """§2 — display-name editing. Collections and the markdown file name stay untouched."""
    name_col, button_col = st.columns([4, 1])
    with name_col:
        new_name = st.text_input("Model name", value=display_name, key=f"edit_model_name_{model_name}")
    with button_col:
        st.write("")  # keep the save button on the input's baseline
        save = st.button("💾 Save name", key=f"save_model_name_{model_name}")

    if save:
        new_name = new_name.strip()
        others = {(model_display_name(other) or other) for other in list_models() if other != model_name}
        if not new_name:
            st.warning("The model name cannot be empty.")
        elif new_name in others:
            st.warning(f"Another model is already named \"{new_name}\".")
        else:
            try:
                update_model_display_name(model_name, new_name)
            except ValueError as error:
                st.warning(str(error))
            else:
                st.session_state["details_message"] = "Model name updated."
                st.rerun()


def _render_description_editor(model_name: str, content: str) -> None:
    """§2 — description editing; an empty save never erases the recorded description."""
    match = re.search(r"\*\*Description:\*\* (.*)", content)
    current = match.group(1).strip() if match else ""
    if current == "*(none provided)*":
        current = ""

    new_description = st.text_area("Model description", value=current,
                                   key=f"edit_model_description_{model_name}")
    if st.button("💾 Save description", key=f"save_model_description_{model_name}"):
        if not new_description.strip():
            st.info("Description cannot be emptied; the previous description was kept.")
        else:
            update_model_description(model_name, new_description.strip())
            st.session_state["details_message"] = "Model description updated."
            st.rerun()


def _render_file_entries(model_name: str) -> None:
    """§3 + §5 — each entry: read-only auto-generated details, an editable description, and a delete button."""
    st.subheader("Files")

    entries = read_model_entries(model_name)
    if not entries:
        st.caption("No files recorded yet — add some below.")
        return

    pending_delete = st.session_state.get("pending_file_delete")

    # Occurrence number per file name — re-uploads create separate entries with
    # the same name, and the occurrence-keyed widgets keep their state when an
    # unrelated entry is deleted. remove_file_entry always removes the most
    # recent entry for a name, so the confirm row shows under that one only.
    totals = Counter(entry["file_name"] for entry in entries)
    occurrences = {}
    for entry in entries:
        occurrence = occurrences.get(entry["file_name"], 0)
        occurrences[entry["file_name"]] = occurrence + 1
        key_id = f"{model_name}_{entry['file_name']}_{occurrence}"
        is_last_occurrence = occurrence + 1 == totals[entry["file_name"]]

        with st.container(border=True):
            if entry["source_type"] == "table" and entry.get("tables"):
                # Table entries render each table as its own section — its
                # details, a scrollable sample dataframe, and its description
                # input right there (§3), not clubbed at the bottom.
                st.markdown(_entry_markdown(_entry_prefix(entry)))
                for t_index, table in enumerate(entry["tables"]):
                    st.markdown(_table_markdown(table))
                    _render_table_sample(table)
                    description = st.text_input(f"Description — {table['table_name']}",
                                                value=table["description"],
                                                key=f"table_description_{key_id}_{t_index}")
                    if st.button("💾 Save description",
                                 key=f"save_table_description_{key_id}_{t_index}"):
                        update_table_description(model_name, entry["file_name"],
                                                 table["table_name"], description.strip())
                        st.session_state["details_message"] = (
                            f"Description saved for {table['table_name']} ({entry['file_name']})."
                        )
                        st.rerun()
                    st.divider()
            else:
                st.markdown(_entry_markdown(entry["text"]))
                description = st.text_input("Description", value=entry["description"],
                                            key=f"file_description_{key_id}")
                if st.button("💾 Save description", key=f"save_file_description_{key_id}"):
                    update_file_description(model_name, entry["file_name"], description.strip())
                    st.session_state["details_message"] = f"Description saved for {entry['file_name']}."
                    st.rerun()

            if st.button("🗑️ Delete file", key=f"delete_file_{key_id}"):
                st.session_state["pending_file_delete"] = entry["file_name"]
                st.rerun()

            if pending_delete == entry["file_name"] and is_last_occurrence:
                st.warning(f"Delete {entry['file_name']} and all of its stored data?")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Confirm delete", key=f"confirm_delete_file_{key_id}", type="primary"):
                    summary = remove_file_entry(model_name, entry["file_name"],
                                                vector_db_path=VECTOR_DB_PATH)
                    st.session_state.pop("pending_file_delete", None)
                    st.session_state["details_message"] = (
                        f"Deleted {entry['file_name']}: {summary.get('documents_deleted', 0)} document(s), "
                        f"{summary.get('files_removed', 0)} file(s) removed."
                    )
                    st.rerun()
                if cancel_col.button("Cancel", key=f"cancel_delete_file_{key_id}"):
                    st.session_state.pop("pending_file_delete", None)
                    st.rerun()


def _entry_markdown(text: str) -> str:
    """Auto-generated details for display — Description lines are edited via the input below."""
    lines = [line for line in text.splitlines()
             if not line.lstrip().startswith("- **Description:**")]
    return "\n".join(lines).strip()


def _entry_prefix(entry: dict) -> str:
    """A table entry's own lines — everything before its first table sub-block."""
    first_table = re.search(r"^  #### ", entry["text"], flags=re.MULTILINE)
    return entry["text"][:first_table.start()] if first_table else entry["text"]


def _table_markdown(table: dict) -> str:
    """A table sub-block's details for display — its Description line and the markdown
    sample table are dropped (the sample renders as a dataframe right below)."""
    lines = [line for line in table["text"].splitlines()
             if not line.lstrip().startswith("- **Description:**")
             and not line.lstrip().startswith("|")]
    return "\n".join(lines).strip()


def _render_table_sample(table: dict) -> None:
    """The recorded 5-row sample as a native dataframe — scrolls horizontally and
    vertically as needed, instead of an overflowing markdown table."""
    match = re.search(r"\*\*Csv file path:\*\* (\S+)", table["text"])
    if not match:
        return
    try:
        dataframe = pd.read_csv(match.group(1)).head(5)
    except Exception:
        st.caption("(sample unavailable)")
        return
    st.dataframe(dataframe, hide_index=True)


def _render_add_files(model_name: str, display_name: str) -> None:
    """§4 — the same uploader and dispatch as the create form, targeting the selected model."""
    with st.expander("➕ Add files to this model"):
        st.caption(f"Files are added to: {display_name}")
        uploaded_files = st.file_uploader("Upload your data files", type=UPLOADER_TYPES,
                                          accept_multiple_files=True,
                                          key=f"add_files_uploader_{model_name}")
        if st.button("Add files", key=f"add_files_submit_{model_name}"):
            if not uploaded_files:
                st.info("No files were selected; nothing to add.")
                return

            records = ingest_uploaded_files(uploaded_files, model_name,
                                            status_label="Adding Files .....",
                                            done_label="Files added!!")
            try:
                # An empty description keeps the recorded one (append semantics).
                data_model_creation(model_name=model_name, description="",
                                    ingestion_records=records)
            except Exception as error:
                st.warning(f"Failed to update model record: {error}")
            else:
                st.session_state["details_message"] = f"Added {len(records)} file(s) to {display_name}."
                st.rerun()


def _render_delete_model(model_name: str, display_name: str) -> None:
    """§6 — whole-model deletion, two-step confirmed."""
    st.divider()

    confirm_key = f"confirm_model_delete_{model_name}"
    if st.button("🗑️ Delete model", key=f"delete_model_{model_name}"):
        st.session_state[confirm_key] = True
        st.rerun()

    if st.session_state.get(confirm_key):
        st.warning(
            f"Delete the entire model {display_name}? This removes its collections, its stored "
            "files, and its record — this cannot be undone."
        )
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm delete", key=f"confirm_delete_model_{model_name}", type="primary"):
            summary = delete_model(model_name, vector_db_path=VECTOR_DB_PATH)
            st.session_state.pop(confirm_key, None)
            st.session_state.pop("selected_model", None)
            st.session_state["create_view_message"] = (
                f"Deleted model {display_name}: {len(summary['collections_deleted'])} collection(s), "
                f"{summary['files_removed']} file(s), "
                f"{'record removed' if summary['description_removed'] else 'no record found'}."
            )
            if summary.get("errors"):
                st.session_state["create_view_warning"] = (
                    "Some parts of the model could not be deleted: " + "; ".join(summary["errors"])
                )
            st.rerun()
        if cancel_col.button("Cancel", key=f"cancel_delete_model_{model_name}"):
            st.session_state.pop(confirm_key, None)
            st.rerun()
