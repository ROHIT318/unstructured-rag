"""
Left pane of the Data Upload page (specs/data_model_nav.md, and the highlight
of specs/data_model_editing.md §7): the Create New Data Model button and the
existing data models, listed by their display name when a record exists. The
model details themselves render in the main area (utils/data_model_details.py)
— only when a model button is clicked here.
"""

import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv

from utils.data_pipeline import sanitize_collection_name
from utils.data_model_creation import model_display_name, recorded_models

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH") or "data/vector_db/"
SIDEBAR_CSS_PATH = Path(__file__).parent / "data_model_sidebar.css"


def model_collection_counts(model_name: str) -> dict:
    """Best-effort document counts of a model's text and image collections."""
    sanitized = sanitize_collection_name(model_name)
    counts = {}
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    for label, collection_name in (("text", sanitized), ("images", f"{sanitized}_images")):
        try:
            counts[label] = client.get_collection(collection_name).count()
        except Exception:
            continue
    return counts


def list_models(vector_db_path: str = VECTOR_DB_PATH) -> list[str]:
    """
    Names of the data models that exist in the vector DB.
    ----
    A collection named `<model>_images` belongs to `<model>` — it is never a model of
    its own. Entries are deduped and sorted alphabetically; every collection counts,
    nothing is filtered out. An unopenable DB yields an empty list. Models with a
    record file but no collections (e.g. created with no files uploaded) are included
    too, so they stay reachable and deletable.
    """
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        collection_names = [collection.name for collection in client.list_collections()]
    except Exception:
        collection_names = []
    models = {name[: -len("_images")] if name.endswith("_images") else name
              for name in collection_names}
    models.update(recorded_models())
    return sorted(models)


def _inject_sidebar_css() -> None:
    """Selected-model highlight styles — kept in data_model_sidebar.css, not inline strings."""
    try:
        css = SIDEBAR_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar() -> None:
    """
    The Create New Data Model button and the existing models. Selecting a model
    stores its underlying name in session state — the main area renders the
    details view on the same run. The selected model's button is the sidebar's
    only primary button, which data_model_sidebar.css recolors (§7).
    """
    with st.sidebar:
        _inject_sidebar_css()

        if st.button("➕ Create New Data Model"):
            st.session_state.pop("selected_model", None)
            st.rerun()

        st.subheader("Existing Data Models")

        models = list_models()
        if not models:
            st.caption("No data models yet — create one!")
            return

        selected_model = st.session_state.get("selected_model")
        for model_name in models:
            display_name = model_display_name(model_name) or model_name
            is_selected = model_name == selected_model
            clicked = st.button(display_name, key=f"model_button_{model_name}",
                                type="primary" if is_selected else "secondary")
            if clicked and not is_selected:
                st.session_state["selected_model"] = model_name
                # Rerun so the clicked button itself renders as the highlighted
                # (primary) one on this same click, not one interaction later.
                st.rerun()
