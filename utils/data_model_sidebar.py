import streamlit as st
import chromadb
import os
from dotenv import load_dotenv

from utils.data_pipeline import sanitize_collection_name

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")

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
    nothing is filtered out. An unopenable DB yields an empty list.
    """
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        collection_names = [collection.name for collection in client.list_collections()]
    except Exception:
        return []
    models = {name[: -len("_images")] if name.endswith("_images") else name
              for name in collection_names}
    return sorted(models)

def render_sidebar() -> None:
    """
    
    """
    with st.sidebar:
        if st.button("➕ Create New Data Model"):
            st.session_state.pop("selected_model", None)

        st.subheader("Existing Data Models")

        models = list_models()
        if not models:
            st.caption("No data models yet — create one!")
            return

        for model_name in models:
            if st.button(model_name):
                st.session_state["selected_model"] = model_name
                render_model_view(model_name=model_name)


def render_model_view(model_name: str=None) -> None:
    """Read-only view of a model's description, with a fallback when it has none."""
    content = read_model_description(model_name)

    if content is not None:
        st.markdown(content)
        return

    st.header(model_name)
    st.markdown("**Description:** *(no description recorded)*")
    counts = model_collection_counts(model_name)
    if "text" in counts:
        st.write(f"Text collection: {counts['text']} document(s)")
    if "images" in counts:
        st.write(f"Image collection: {counts['images']} document(s)")


def read_model_description():
    pass


def model_collection_counts():
    pass