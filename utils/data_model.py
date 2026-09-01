import streamlit as st

from utils.data_model_creation import data_model_creation
from utils.data_model_details import render_details_view
from utils.data_model_ingestion import ingest_uploaded_files
from utils.data_model_sidebar import render_sidebar


def render_create_view() -> None:
    """The upload form — unchanged fields, dispatch, and messages."""
    message = st.session_state.pop("create_view_message", None)
    warning = st.session_state.pop("create_view_warning", None)
    if message:
        st.success(message)
    if warning:
        st.warning(warning)

    with st.form("model_form"):
        uploaded_file = st.file_uploader(
            "Upload your data files",
            type=["csv", "xlsx", "xlsm", "xlsb", "txt", "jpeg", "jpg", "png", "pdf", "pptx"],
            accept_multiple_files=True
        )
        model_name = st.text_input("Model Name")
        model_description = st.text_input("model_description")
        submitted = st.form_submit_button("Submit")

        if submitted:
            ingestion_records = ingest_uploaded_files(uploaded_file, model_name)

            try:
                record_path = data_model_creation(model_name=model_name, description=model_description,
                                                  ingestion_records=ingestion_records)
                # The sidebar rendered before this submission was processed, so
                # it still lists the pre-creation models — rerun (with the
                # success message carried in session state) to refresh it and
                # show the new model without a manual page refresh.
                st.session_state["create_view_message"] = f"Model record updated: {record_path}"
                st.rerun()
            except Exception as e:
                st.warning(f"Failed to write model description: {e}")


render_sidebar()

selected_model = st.session_state.get("selected_model")
if selected_model:
    render_details_view(selected_model)
else:
    render_create_view()

# Section 2: Add relationship option: Manual or autmatic relationship creation and end user getting the option to add or remove it

# Section 3: Data Details - Table or unstructured data in dropdown; Data View (tabular or data view); data description (input text box); update button
