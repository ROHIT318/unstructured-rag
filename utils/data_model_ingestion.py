"""
Shared upload dispatch for the Data Upload page — used by both the create form
(utils/data_model.py) and the add-files expander of the model details view
(utils/data_model_details.py). Writes each uploaded file to a temp/permanent
path, dispatches it by extension to the matching store_* pipeline function
(utils/data_pipeline.py), and collects one ingestion record per file for the
model's markdown record (utils/data_model_creation.py).
"""

import os
import uuid

import streamlit as st

from utils.data_pipeline import store_image, store_pdf, store_ppt, store_table, store_txt


def ingest_uploaded_files(uploaded_files, model_name: str,
                          status_label: str = "Model Creation In Progress .....",
                          done_label: str = "Model creation completed!!") -> list[dict]:
    """
    Run every uploaded file through the ingestion pipeline and return the
    ingestion records for the model's markdown file. Ingestion status messages
    are emitted from here directly, matching the pipeline's existing style.
    """
    ingestion_records = []

    with st.status(status_label, expanded=False) as status:
        for file in uploaded_files or []:
            try:
                st.badge(f"Processing file {file.name}", icon=":material/check:", color="green")

                if file.name.endswith("pdf"):
                    temp_file_name = f"data/vector_db/temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_pdf(path=temp_file_name, model_name=model_name,
                                       file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "pdf",
                        "collection_name": result["collection_name"],
                        "document_ids": result["document_ids"],
                        "images": result["images"],
                    })

                elif file.name.endswith(("jpeg", "jpg", "png")):
                    temp_file_name = f"data/images/{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_image(path=temp_file_name, model_name=model_name,
                                         file_name=file.name, return_ids=True)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "image",
                        "collection_name": result["collection_name"],
                        "file_path": result["file_path"],
                        "document_ids": result["document_ids"],
                    })

                elif file.name.endswith("pptx"):
                    temp_file_name = f"data/vector_db/temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_ppt(path=temp_file_name, model_name=model_name,
                                       file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "ppt",
                        "collection_name": result["collection_name"],
                        "document_ids": result["document_ids"],
                        "images": result["images"],
                    })

                elif file.name.endswith(("csv", "xlsx", "xlsm", "xlsb")):
                    temp_file_name = f"data/vector_db/temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_table(path=temp_file_name, model_name=model_name,
                                         file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "table",
                        "collection_name": result["collection_name"],
                        "tables": result["tables"],
                    })

                elif file.name.endswith("txt"):
                    temp_file_name = f"data/vector_db/temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_txt(path=temp_file_name, model_name=model_name,
                                       file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "txt",
                        "collection_name": result["collection_name"],
                        "document_ids": result["document_ids"],
                    })

                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    result = f"Skipped {file.name}: unsupported file type"
                    ingestion_records.append({"file_name": file.name, "source_type": "skipped"})

                st.write(result)
            except Exception as e:
                st.write(f"Failed to process file with error: {e}")
                continue
            st.divider()

        status.update(label=done_label, state="complete", expanded=False)

    return ingestion_records
