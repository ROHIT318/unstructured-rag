from pypdf import PdfReader
from dotenv import load_dotenv
import os
from typing import List
import uuid
import streamlit as st

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb


from google import genai
from google.genai import types

load_dotenv()

GEMINI_EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")

embedding_model = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GEMINI_EMBEDDING_API_KEY)
client = genai.Client(api_key=GEMINI_EMBEDDING_API_KEY)


def chunked_documents(text_content: str, chunking_method: str = "text_structure_based"):
    split_documents = ""

    if chunking_method=="semantic_chunker":
        # text_content = text_content.replace(".", " ")
        text_splitter = SemanticChunker(
            embeddings=embedding_model, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.3
        )
        split_documents = text_splitter.create_documents([text_content])

    elif chunking_method=="text_structure_based":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_documents = text_splitter.split_text(text_content)

    else:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        split_documents = text_splitter.split_text(text_content)

    return split_documents


def store_pdf(path: str, model_name: str = str(uuid.uuid4()), chunking_method: str ="text_structure_based"):
    """
    Arg:
        path: path to the pdf file.
        chunking_method: What type of chunking do you want to apply on your pdf text content. Possible inputs are ["text_structure_based", "markdown_based", "semantic_chunker"]
    
    """
    reader = PdfReader(path)
    text_content = ""

    for page in reader.pages:
        text_content += page.extract_text()

    documents = chunked_documents(text_content, chunking_method=chunking_method)

    if documents is not None: 
        st.write("Chunking of PDF completed.....")
    else:
        st.write("Chunking issue with PDF.....!!")
        return

    persistent_collection, ids = store_in_vector_db(documents=documents, model_name=model_name)
    if persistent_collection and ids:
        st.write("PDF data stored in vector database.....")
    else:
        st.write("PDF data was not stored in vector database.....!!")

    return "Storing PDF operation completed....."


def store_in_vector_db(documents: List[str], model_name: str, collection_path: str = "data/vector_db/", embeddings: str = None):
        
    persistent_client = chromadb.PersistentClient(path=collection_path)
    persistent_collection = persistent_client.get_or_create_collection(model_name)
    st.write(f"Created collection {model_name}")

    ids = []
    for i in range(0, len(documents)):
        ids.append(str(uuid.uuid4()))

    if embeddings is None:
        persistent_collection.add(documents=documents, ids=ids)
    else:
        persistent_collection.add(documents=documents, embeddings=embeddings, ids=ids)

    return persistent_collection, ids


def get_image_embedding(image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
    )
    return result.embeddings[0].values

def store_image(path: str, model_name: str = None): 
    image_bytes = b""

    if path is None:
        return

    if model_name is None:
        model_name = str(uuid.uuid4())

    with open(path, "rb") as f:
        image_bytes = f.read()

    mime_type = "image/png" if path.endswith(".png") else "image/jpeg"
    file_name = path.split('/')[-1]

    vector = get_image_embedding(image_bytes, mime_type=mime_type)

    st.write(f"Created embeddings for image {file_name}.....")
    result = store_in_vector_db(documents=[path], model_name=model_name, embeddings=[vector])
    st.write(f"Stored embeddings for image {file_name}.....")
    
    return result


if __name__ == "__main__":
    # chunk_method = ["text_structure_based", "markdown_based", "semantic_chunker"]
    # pdf_content = store_pdf(path="data/vector_db/bkn_duronto.pdf", chunking_method=chunk_method[1])

    # for content in pdf_content:
    #     print(content, end="\n\n")
    # print(pdf_content)


    result = store_image(path="data/images/ebe60544-cb00-4010-a0eb-1c082b53389a_resume_rohit_sharma.jpg", model_name="test_model")
    print(type(result))
    print(result)