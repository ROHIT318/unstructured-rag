from pathlib import Path

import streamlit as st

from utils.data_model_creation import model_display_name, recorded_models
from utils.neo4j_ingestion import get_model_names

SIDEBAR_CSS_PATH = Path(__file__).parent / "data_model_sidebar.css"


def list_models() -> list[str]:
    """
    Names of the data models that exist — the Model nodes in Neo4j, unioned
    with every model that has a record file (records created before the Neo4j
    switch have no Model node; they stay reachable and deletable through this
    union). Entries are deduped and sorted alphabetically; an unreachable
    database yields the record-only list.
    """
    try:
        models = set(get_model_names())
    except Exception:
        models = set()
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
