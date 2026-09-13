import os
import uuid

import streamlit as st

from utils.data_model_creation import sanitize_model_name
from utils.data_pipeline import store_image, store_pdf, store_ppt, store_table, store_txt
from utils.neo4j_ingestion import ensure_model_node

TEMP_FOLDER = "data/temp/"


def ingest_uploaded_files(uploaded_files, model_name: str, model_description: str = "",
                          status_label: str = "Model Creation In Progress .....",
                          done_label: str = "Model creation completed!!") -> list[dict]:
    """
    Run every uploaded file through the ingestion pipeline and return the
    ingestion records for the model's markdown file. The model's Model node is
    created (or, for add-files, reused) up front with the given description —
    an empty description never overwrites the recorded one. Ingestion status
    messages are emitted from here directly, matching the pipeline's existing
    style.
    """
    ingestion_records = []
    underlying_name = sanitize_model_name(model_name)
    ensure_model_node(underlying_name, model_description)

    with st.status(status_label, expanded=False) as status:
        for file in uploaded_files or []:
            try:
                st.badge(f"Processing file {file.name}", icon=":material/check:", color="green")

                if file.name.endswith("pdf"):
                    temp_file_name = f"{TEMP_FOLDER}temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_pdf(path=temp_file_name, model_name=underlying_name,
                                       file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "pdf",
                        "file_id": result["file_id"],
                        "chunk_ids": result["chunk_ids"],
                        "images": result["images"],
                    })

                elif file.name.endswith(("jpeg", "jpg", "png")):
                    # Image uploads persist directly — the path is the stored pointer.
                    temp_file_name = f"data/images/{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_image(path=temp_file_name, model_name=underlying_name,
                                         file_name=file.name, return_ids=True)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "image",
                        "file_id": result["file_id"],
                        "file_path": result["file_path"],
                        "image_id": result["image_id"],
                    })

                elif file.name.endswith("pptx"):
                    temp_file_name = f"{TEMP_FOLDER}temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_ppt(path=temp_file_name, model_name=underlying_name,
                                       file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "ppt",
                        "file_id": result["file_id"],
                        "chunk_ids": result["chunk_ids"],
                        "images": result["images"],
                    })

                elif file.name.endswith(("csv", "xlsx", "xlsm", "xlsb")):
                    temp_file_name = f"{TEMP_FOLDER}temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_table(path=temp_file_name, model_name=underlying_name,
                                         file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "table",
                        "file_id": result["file_id"],
                        "tables": result["tables"],
                    })

                elif file.name.endswith("txt"):
                    temp_file_name = f"{TEMP_FOLDER}temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_txt(path=temp_file_name, model_name=underlying_name,
                                       file_name=file.name, return_ids=True)
                    os.remove(temp_file_name)
                    ingestion_records.append({
                        "file_name": result["file_name"],
                        "source_type": "txt",
                        "file_id": result["file_id"],
                        "chunk_ids": result["chunk_ids"],
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
