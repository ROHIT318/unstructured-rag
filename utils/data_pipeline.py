from pypdf import PdfReader
from dotenv import load_dotenv
import os
import re
from typing import List
import uuid
import streamlit as st

import pandas as pd

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


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


def sanitize_collection_name(model_name: str) -> str:
    """
    Chroma only accepts collection names with 3-512 characters from
    [a-zA-Z0-9._-], starting and ending with an alphanumeric character.
    Anything else (e.g. spaces) is replaced with underscores, and names that
    are left too short get a "model_" prefix so the mapping from model name to
    collection stays deterministic.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", str(model_name))
    sanitized = sanitized[:512].strip("._-")

    if len(sanitized) < 3:
        sanitized = ("model_" + sanitized).strip("._-")

    return sanitized


def store_in_vector_db(documents: List[str], model_name: str, collection_path: str = "data/vector_db/", embeddings: str = None):

    collection_name = sanitize_collection_name(model_name)
    persistent_client = chromadb.PersistentClient(path=collection_path)
    persistent_collection = persistent_client.get_or_create_collection(collection_name)
    st.write(f"Created collection {collection_name}")

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
    # Images live in their own collection: Gemini image embeddings (3072-dim)
    # cannot share a collection with text documents, which use Chroma's
    # default 384-dim embedder.
    result = store_in_vector_db(documents=[path], model_name=f"{model_name}_images", embeddings=[vector])
    st.write(f"Stored embeddings for image {file_name}.....")

    return result


def extract_ppt_content(path: str):
    """
    Walk every slide of a pptx file and pull out its text and images.

    Arg:
        path: path to the pptx file.

    Returns:
        text_content: all slide text (shapes, tables, notes) joined together.
        images: list of {"bytes": ..., "mime_type": ..., "ext": ...} for every embedded picture.
    """
    presentation = Presentation(path)
    text_parts = []
    images = []

    def walk_shapes(shapes, slide_index):
        for shape in shapes:
            # Groups nest other shapes, recurse into them.
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                walk_shapes(shape.shapes, slide_index)
                continue

            # Tables: read the text cell by cell, row by row.
            if shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        text_parts.append(row_text)
                continue

            # Regular text (title, body, text boxes).
            if shape.has_text_frame:
                shape_text = shape.text_frame.text.strip()
                if shape_text:
                    text_parts.append(shape_text)

            # Pictures, including picture placeholders. Linked (not embedded)
            # images raise ValueError, so guard the access.
            try:
                image = shape.image
            except (AttributeError, ValueError):
                image = None

            if image is not None:
                images.append({
                    "bytes": image.blob,
                    "mime_type": image.content_type,
                    "ext": image.ext,
                })

    for slide_index, slide in enumerate(presentation.slides):
        walk_shapes(slide.shapes, slide_index)

        # Speaker notes, when present.
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame is not None and notes_frame.text.strip():
                text_parts.append(notes_frame.text.strip())

    text_content = "\n\n".join(text_parts)
    return text_content, images


def store_ppt(path: str, model_name: str = str(uuid.uuid4()), chunking_method: str = "text_structure_based"):
    """
    Extract text and images from a pptx file and store them in the vector database.

    Slide text is chunked and stored in a text collection, while slide images are
    embedded with Gemini and stored in a separate image collection (Chroma pins each
    collection to a single embedding dimension, so text and images cannot share one).

    Arg:
        path: path to the pptx file.
        model_name: name of the model/collection the data belongs to.
        chunking_method: chunking to apply on the slide text. Possible inputs are ["text_structure_based", "markdown_based", "semantic_chunker"]
    """
    text_content, images = extract_ppt_content(path)
    file_name = path.split('/')[-1]
    image_folder = "data/images/"

    st.write(f"Extracted {len(images)} image(s) from PPT {file_name}.....")

    # Store slide text.
    if text_content.strip():
        documents = chunked_documents(text_content, chunking_method=chunking_method)

        if documents is not None:
            st.write("Chunking of PPT text completed.....")
        else:
            st.write("Chunking issue with PPT.....!!")
            return

        persistent_collection, ids = store_in_vector_db(documents=documents, model_name=model_name)
        if persistent_collection and ids:
            st.write("PPT text stored in vector database.....")
        else:
            st.write("PPT text was not stored in vector database.....!!")
    else:
        st.write("No text content found in PPT.....")

    # Store slide images in a separate image collection.
    if images:
        image_documents = []
        image_embeddings = []

        for image in images:
            # Keep the picture on disk so retrieval can load it back later, the same
            # way a directly uploaded image is kept.
            image_path = f"{image_folder}{uuid.uuid4()}.{image['ext']}"
            with open(image_path, "wb") as f:
                f.write(image["bytes"])

            image_documents.append(image_path)
            image_embeddings.append(get_image_embedding(image["bytes"], mime_type=image["mime_type"]))

        persistent_collection, ids = store_in_vector_db(
            documents=image_documents,
            model_name=f"{model_name}_images",
            embeddings=image_embeddings,
        )
        if persistent_collection and ids:
            st.write("PPT images stored in vector database.....")
        else:
            st.write("PPT images were not stored in vector database.....!!")

    return "Storing PPT operation completed....."


def extract_table_content(path: str):
    """
    Read every table of a csv/xlsx/xlsm/xlsb file.

    Arg:
        path: path to the file.

    Returns:
        tables: list of {"sheet_name": ..., "dataframe": ...} for every
        non-empty table. csv files yield a single entry with sheet_name None;
        Excel workbooks yield one entry per non-empty sheet.
    """
    if path.endswith("csv"):
        sheets = {None: pd.read_csv(path)}
    else:
        engine = "pyxlsb" if path.endswith("xlsb") else "openpyxl"
        sheets = pd.read_excel(path, engine=engine, sheet_name=None)

    tables = []
    for sheet_name, dataframe in sheets.items():
        # Skip sheets that have no columns or no rows at all.
        if dataframe.empty:
            continue
        tables.append({"sheet_name": sheet_name, "dataframe": dataframe})

    return tables


def store_table(path: str, model_name: str = str(uuid.uuid4())):
    """
    Save csv/excel tables as csv files and register them in the vector database.

    The table content itself is not embedded. Each table is written to
    data/tables/ as a csv, and a small details document (path, sheet name,
    columns, row count) is stored in the model's text collection so retrieval
    knows the table exists and can fetch it from disk — the same pointer
    pattern used for images.

    Arg:
        path: path to the csv/xlsx/xlsm/xlsb file.
        model_name: name of the model/collection the data belongs to.
    """
    tables = extract_table_content(path)
    file_name = os.path.basename(path)
    file_stem = re.sub(r"[^\w\-. ]", "_", file_name.rsplit(".", 1)[0])
    table_folder = "data/tables/"

    st.write(f"Extracted {len(tables)} table(s) from {file_name}.....")

    table_documents = []
    for table in tables:
        # One csv per table; Excel sheets are suffixed with the sheet name.
        if table["sheet_name"] is None:
            csv_name = f"{uuid.uuid4()}_{file_stem}.csv"
        else:
            sheet_name = re.sub(r"[^\w\-. ]", "_", str(table["sheet_name"]))
            csv_name = f"{uuid.uuid4()}_{file_stem}__{sheet_name}.csv"

        csv_path = f"{table_folder}{csv_name}"
        table["dataframe"].to_csv(csv_path, index=False)

        dataframe = table["dataframe"]
        sheet_line = f" (Sheet: {table['sheet_name']})" if table["sheet_name"] is not None else ""
        table_documents.append(
            f"Table: {file_name}{sheet_line}\n"
            f"Path: {csv_path}\n"
            f"Columns: {', '.join(str(column) for column in dataframe.columns)}\n"
            f"Rows: {len(dataframe)}"
        )

    if table_documents:
        persistent_collection, ids = store_in_vector_db(documents=table_documents, model_name=model_name)
        if persistent_collection and ids:
            st.write("Table details stored in vector database.....")
        else:
            st.write("Table details were not stored in vector database.....!!")
    else:
        st.write("No tables found in file.....")

    return "Storing table operation completed....."


def store_txt(path: str, model_name: str = str(uuid.uuid4()), chunking_method: str = "text_structure_based"):
    """
    Arg:
        path: path to the txt file.
        chunking_method: What type of chunking do you want to apply on your txt text content. Possible inputs are ["text_structure_based", "markdown_based", "semantic_chunker"]

    """
    with open(path, "r", encoding="utf-8") as f:
        text_content = f.read()

    documents = chunked_documents(text_content, chunking_method=chunking_method)

    if documents is not None:
        st.write("Chunking of TXT completed.....")
    else:
        st.write("Chunking issue with TXT.....!!")
        return

    persistent_collection, ids = store_in_vector_db(documents=documents, model_name=model_name)
    if persistent_collection and ids:
        st.write("TXT data stored in vector database.....")
    else:
        st.write("TXT data was not stored in vector database.....!!")

    return "Storing TXT operation completed....."


if __name__ == "__main__":
    # chunk_method = ["text_structure_based", "markdown_based", "semantic_chunker"]
    # pdf_content = store_pdf(path="data/vector_db/bkn_duronto.pdf", chunking_method=chunk_method[1])

    # for content in pdf_content:
    #     print(content, end="\n\n")
    # print(pdf_content)


    result = store_image(path="data/images/ebe60544-cb00-4010-a0eb-1c082b53389a_resume_rohit_sharma.jpg", model_name="test_model")
    print(type(result))
    print(result)