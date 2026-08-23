import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

st.title("📚 Unstructured RAG Application")

st.write(
    """
    Welcome to the Unstructured RAG application.

    This application uses Retrieval-Augmented Generation (RAG) to work with
    unstructured documents such as PDFs, text files, reports, and other
    document formats. The documents are processed, converted into searchable
    chunks, and used as context for generating relevant answers.
    """
)

st.divider()

# Description
st.header("📖 Description")

st.write(
    """
    Unstructured RAG allows users to ask questions about information stored
    in unstructured documents.

    Instead of relying only on the knowledge of a Large Language Model (LLM),
    the application first retrieves relevant information from the uploaded
    documents and then provides that information to the LLM to generate an
    answer.
    """
)

# Details
st.header("⚙️ Key Details")

st.markdown(
    """
    - **Document Ingestion:** Upload or provide unstructured documents.
    - **Document Processing:** Extract and clean the document content.
    - **Chunking:** Split documents into smaller meaningful sections.
    - **Embeddings:** Convert document chunks into numerical vectors.
    - **Vector Database:** Store embeddings for efficient searching.
    - **Retrieval:** Find relevant document chunks based on the user's query.
    - **LLM:** Use the retrieved information to generate an answer.
    """
)

# How it works
st.header("🔄 How It Works")

st.code(
    """
Documents
    ↓
Document Processing
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
    ↓
User Query
    ↓
Retrieve Relevant Chunks
    ↓
LLM + Retrieved Context
    ↓
Final Answer
""",
    language="text"
)

# Instructions
st.header("📝 Working Instructions")

st.markdown(
    """
    1. Upload or provide the required documents.
    2. Process the documents to extract their content.
    3. Split the content into appropriate chunks.
    4. Generate embeddings for the chunks.
    5. Store the embeddings in the vector database.
    6. Enter a question related to the uploaded documents.
    7. The application retrieves the most relevant information.
    8. The retrieved information is provided to the LLM.
    9. The LLM generates the final response.
    """
)

# Technologies
st.header("🛠️ Technologies")

st.markdown(
    """
    - Python
    - Streamlit
    - Unstructured
    - Embedding Model
    - Vector Database
    - Large Language Model (LLM)
    """
)

st.divider()

st.caption("Unstructured RAG Application | Built with Streamlit")