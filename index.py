import streamlit as st
import os

# create folders required for functionality
MEDIA_FOLDER_NAME_LS = ["data", "data/images", "data/vector_db", "data/tables", "data/conversion_history"]
for media_folder_name in MEDIA_FOLDER_NAME_LS: 
    if not os.path.exists(media_folder_name):
        os.makedirs(media_folder_name)

home_page = st.Page("utils/home.py", title="Home", icon="🏠", default=True)
data_upload_page = st.Page("utils/data_model.py", title="Data Upload", icon="📊")
chat_assistant_page = st.Page("utils/chat_model.py", title="Chat Assistant", icon="⚙️")

nav = st.navigation([home_page, data_upload_page, chat_assistant_page], position="top")

if __name__ == "__main__":
    nav.run()

