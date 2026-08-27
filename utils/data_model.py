import streamlit as st
import os
from utils.data_pipeline import store_pdf, store_image, store_ppt
import uuid

# Section 1: File upload component, model name, model description, submit button
with st.form("model_form"):
    uploaded_file = st.file_uploader(
        "Upload your data files", 
        type=["csv", "xlsx", "xlsm", "xlsb", "jpeg", "jpg", "png", "pdf", "pptx"], 
        accept_multiple_files=True
    )
    model_name = st.text_input("Model Name")
    model_description = st.text_input("model_description")
    submitted = st.form_submit_button("Submit")

    if submitted:

        file_upload_location = []

        # with st.spinner(text="Model Creation In Progress .....", show_time=True):
        with st.status("Model Creation In Progress .....", expanded=False) as status:
            for file in uploaded_file:
                st.write(f"Processing file {file.name}" )

                if file.name.endswith("pdf"):
                    temp_file_name = f"data/vector_db/temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_pdf(path=temp_file_name, model_name=model_name)
                    os.remove(temp_file_name)             


                elif file.name.endswith(("jpeg", "jpg", "png")):
                    temp_file_name = f"data/images/{uuid.uuid4()}_{file.name}"
                    image_bytes = ""
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_image(path=temp_file_name, model_name=model_name)


                elif file.name.endswith("pptx"):
                    temp_file_name = f"data/vector_db/temp_{uuid.uuid4()}_{file.name}"
                    with open(temp_file_name, "wb") as f:
                        f.write(file.getbuffer())
                    result = store_ppt(path=temp_file_name, model_name=model_name)
                    os.remove(temp_file_name)


                elif file.name.endswith("csv"):
                    pass

                elif file.name.endswith(("xlsx", "xlsb", "xlsm")):
                    pass


                else:
                    pass

                st.write(result)

            status.update(label="Model creation completed!!", state="complete", expanded=False)

# Section 2: Add relationship option: Manual or autmatic relationship creation and end user getting the option to add or remove it

# Section 3: Data Details - Table or unstructured data in dropdown; Data View (tabular or data view); data description (input text box); update button