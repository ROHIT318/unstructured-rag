from pypdf import PdfReader
from dotenv import load_dotenv
import os
import re
import uuid
import streamlit as st

import pandas as pd

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.img_to_txt import extract_text_from_image
from utils.neo4j_ingestion import create_file_node, store_file_elements, txt_to_embeddings

load_dotenv()

GEMINI_EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
BASE_URL = os.getenv("BASE_URL")

embedding_kwargs = {
    "model": GEMINI_EMBEDDING_MODEL,
    "google_api_key": GEMINI_EMBEDDING_API_KEY,
}
if BASE_URL:
    embedding_kwargs["base_url"] = BASE_URL

embedding_model = GoogleGenerativeAIEmbeddings(**embedding_kwargs)


def enrich_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """
    The one shared image enrichment step, run identically at every image entry
    point (direct upload, PDF page, PPT slide). Produces the image's two texts
    — semantic meaning (genai) and OCR (pytesseract) — plus Gemini text
    embeddings of both at the shared text dimension, so image retrieval is
    text-based and comparable with chunk text. The multimodal image embedding
    path is gone; image bytes are never embedded anymore.

    Returns the Image-element properties:
        semantic_meaning, semantic_embedding, ocr_text, and ocr_embedding —
        the last one omitted entirely when the image has no readable text, so
        such nodes simply stay out of the OCR vector index.

    Extraction failures are already absorbed inside utils/img_to_txt.py (empty
    OCR text / fallback semantic string); embedding failures here propagate,
    like chunk embedding failures do.
    """
    ocr_text, semantic_meaning = extract_text_from_image(image_bytes, mime_type)

    texts_to_embed = [semantic_meaning] + ([ocr_text] if ocr_text else [])  
    embeddings = txt_to_embeddings(texts_to_embed)

    properties = {
        "semantic_meaning": semantic_meaning,
        "semantic_embedding": embeddings[0],
        "ocr_text": ocr_text,
    }
    if ocr_text:
        properties["ocr_embedding"] = embeddings[1]
    return properties


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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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


def store_pdf(path: str, model_name: str = str(uuid.uuid4()), chunking_method: str = "text_structure_based", file_name: str = None, return_ids: bool = False):
    """
    Arg:
        path: path to the pdf file.
        chunking_method: What type of chunking do you want to apply on your pdf text content. Possible inputs are ["text_structure_based", "markdown_based", "semantic_chunker"]
        file_name: original uploaded file name; derived from the path when omitted (uploads land in temp_<uuid>_<name> files).
        return_ids: with True, return a dict with the file's Neo4j node ids (file id, chunk ids, extracted image ids) instead of the status string.

    The file is walked page by page and every page's text is chunked on its own,
    so the elements handed to Neo4j follow the document's true reading order —
    a page's text chunks, then that page's images, then the next page.
    """
    if file_name is None:
        file_name = re.sub(r"^temp_[0-9a-fA-F-]{36}_", "", os.path.basename(path))

    file_id = create_file_node(model_name, file_name, "pdf")
    reader = PdfReader(path)
    file_stem = file_name.rsplit(".", 1)[0]
    elements = []
    pdf_images = []
    text_found = False

    for page in reader.pages:
        page_text = page.extract_text() or ""

        # A page's text chunks come first, in reading order.
        if page_text.strip():
            text_found = True
            for document in chunked_documents(page_text, chunking_method=chunking_method):
                elements.append({"kind": "chunk", "content": document})

        # Then the page's embedded images, kept on disk for retrieval.
        for image in page.images:
            try:
                temp_file_name = f"data/images/{uuid.uuid4()}_{file_stem}.png"
                st.write(temp_file_name)
                with open(temp_file_name, "wb") as f:
                    f.write(image.data)
                elements.append({
                    "kind": "image",
                    "file_name": file_name,
                    "file_path": temp_file_name,
                    **enrich_image(image.data, mime_type="image/png"),
                })
                pdf_images.append(temp_file_name)
            except Exception as e:
                st.warning(f"Failed to store image {image.name}: {e}")

    st.success(f"Extracted {len(pdf_images)} image(s) from PDF.....")

    if not text_found:
        st.warning("No text layer found in PDF — likely scanned, skipping text chunking.....!!")

    stored = store_file_elements(file_id, elements)
    chunk_ids = stored["chunk_ids"]
    images = [{"file_path": file_path, "image_id": image_id}
              for file_path, image_id in zip(pdf_images, stored["image_ids"])]
    if chunk_ids or images:
        st.success("PDF data stored in Neo4j.....")
    else:
        st.warning("PDF data was not stored in Neo4j.....!!")

    if return_ids:
        return {
            "result": "Storing PDF operation completed.....",
            "file_name": file_name,
            "file_id": file_id,
            "chunk_ids": chunk_ids,
            "images": images,
        }

    return "Storing PDF operation completed....."


def store_image(path: str, model_name: str = None, file_name: str = None, return_ids: bool = False, file_id: str = None):
    """
    Arg:
        path: path to the image file.
        model_name: name of the model the image belongs to; only used to create the File node when file_id is omitted.
        file_name: original uploaded file name; derived from the path when omitted (uploads land in <uuid>_<name> files).
        file_id: the File node to attach the image to; direct uploads leave it None and get a File node of their own.
        return_ids: with True, return a dict with the file/image node ids and the file path instead of the status string.
    """
    if path is None:
        return

    if model_name is None:
        model_name = str(uuid.uuid4())

    if file_name is None:
        file_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", os.path.basename(path))

    with open(path, "rb") as f:
        image_bytes = f.read()

    mime_type = "image/png" if path.endswith(".png") else "image/jpeg"

    # The bytes stay on disk in data/images/; the Image node carries the pointer
    # path plus the OCR and semantic-meaning texts with their embeddings.
    properties = enrich_image(image_bytes, mime_type=mime_type)
    st.success(f"Extracted text and embeddings for image {file_name}.....")

    if file_id is None:
        file_id = create_file_node(model_name, file_name, "image")
    stored = store_file_elements(file_id, [{
        "kind": "image",
        "file_name": file_name,
        "file_path": path,
        **properties,
    }])
    st.success(f"Stored image {file_name} in Neo4j.....")

    if return_ids:
        return {
            "result": "Storing image operation completed.....",
            "file_name": file_name,
            "file_id": file_id,
            "file_path": path,
            "image_id": stored["image_ids"][0],
        }

    return "Storing image operation completed....."


def extract_ppt_content(path: str):
    """
    Walk every slide of a pptx file and pull out its text and images.

    Arg:
        path: path to the pptx file.

    Returns:
        slides: list of {"text": ..., "images": [...]} per slide, in slide
        order. Each images entry is {"bytes": ..., "mime_type": ..., "ext": ...}
        for an embedded picture of that slide.
    """
    presentation = Presentation(path)
    slides = []

    def walk_shapes(shapes, slide):
        for shape in shapes:
            # Groups nest other shapes, recurse into them.
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                walk_shapes(shape.shapes, slide)
                continue

            # Tables: read the text cell by cell, row by row.
            if shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        slide["text_parts"].append(row_text)
                continue

            # Regular text (title, body, text boxes).
            if shape.has_text_frame:
                shape_text = shape.text_frame.text.strip()
                if shape_text:
                    slide["text_parts"].append(shape_text)

            # Pictures, including picture placeholders. Linked (not embedded)
            # images raise ValueError, so guard the access.
            try:
                image = shape.image
            except (AttributeError, ValueError):
                image = None

            if image is not None:
                slide["images"].append({
                    "bytes": image.blob,
                    "mime_type": image.content_type,
                    "ext": image.ext,
                })

    for slide in presentation.slides:
        slide_content = {"text_parts": [], "images": []}
        walk_shapes(slide.shapes, slide_content)

        # Speaker notes, when present.
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame is not None and notes_frame.text.strip():
                slide_content["text_parts"].append(notes_frame.text.strip())

        slides.append(slide_content)

    return slides


def store_ppt(path: str, model_name: str = str(uuid.uuid4()), chunking_method: str = "text_structure_based", file_name: str = None, return_ids: bool = False):
    """
    Extract text and images from a pptx file and store them in Neo4j.

    The file is walked slide by slide and every slide's text is chunked on its
    own, so the elements handed to Neo4j follow the document's true reading
    order — a slide's text chunks, then that slide's pictures, then the next
    slide. Slide pictures are kept on disk in data/images/ as Image nodes under
    the same File; slide text is embedded with Gemini as Chunk nodes.

    Arg:
        path: path to the pptx file.
        model_name: name of the model the data belongs to.
        chunking_method: chunking to apply on the slide text. Possible inputs are ["text_structure_based", "markdown_based", "semantic_chunker"]
        file_name: original uploaded file name; derived from the path when omitted (uploads land in temp_<uuid>_<name> files).
        return_ids: with True, return a dict with the file's Neo4j node ids (file id, chunk ids, slide image ids) instead of the status string.
    """
    if file_name is None:
        file_name = re.sub(r"^temp_[0-9a-fA-F-]{36}_", "", os.path.basename(path))

    file_id = create_file_node(model_name, file_name, "ppt")
    slides = extract_ppt_content(path)
    image_folder = "data/images/"
    elements = []
    ppt_images = []

    st.success(f"Extracted {sum(len(slide['images']) for slide in slides)} image(s) from PPT {file_name}.....")

    for slide in slides:
        # A slide's text chunks come first, in reading order.
        slide_text = "\n\n".join(slide["text_parts"])
        if slide_text.strip():
            for document in chunked_documents(slide_text, chunking_method=chunking_method):
                elements.append({"kind": "chunk", "content": document})

        # Then the slide's pictures, kept on disk so retrieval can load them
        # back later, the same way a directly uploaded image is kept.
        for image in slide["images"]:
            image_path = f"{image_folder}{uuid.uuid4()}.{image['ext']}"
            with open(image_path, "wb") as f:
                f.write(image["bytes"])

            elements.append({
                "kind": "image",
                "file_name": file_name,
                "file_path": image_path,
                **enrich_image(image["bytes"], mime_type=image["mime_type"]),
            })
            ppt_images.append(image_path)

    stored = store_file_elements(file_id, elements)
    chunk_ids = stored["chunk_ids"]
    images = [{"file_path": file_path, "image_id": image_id}
              for file_path, image_id in zip(ppt_images, stored["image_ids"])]
    if chunk_ids or images:
        st.success("PPT data stored in Neo4j.....")
    else:
        st.warning("PPT data was not stored in Neo4j.....!!")

    if return_ids:
        return {
            "result": "Storing PPT operation completed.....",
            "file_name": file_name,
            "file_id": file_id,
            "chunk_ids": chunk_ids,
            "images": images,
        }

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


def store_table(path: str, model_name: str = str(uuid.uuid4()), file_name: str = None, return_ids: bool = False):
    """
    Save csv/excel tables as csv files and register them in Neo4j.

    The table content itself is not embedded. Each table is written to
    data/tables/ as a csv, and a Table node (name, csv path, column names,
    column types, row count) is created under the file's File node so retrieval
    knows the table exists and can fetch it from disk — the same pointer
    pattern used for images.

    Arg:
        path: path to the csv/xlsx/xlsm/xlsb file.
        model_name: name of the model the data belongs to.
        file_name: original uploaded file name; derived from the path when omitted (uploads land in temp_<uuid>_<name> files).
        return_ids: with True, return a dict with the file id and one record per extracted table (table id, table name, csv path, top 5 rows, column names, column data types, row count) instead of the status string.
    """
    if file_name is None:
        file_name = re.sub(r"^temp_[0-9a-fA-F-]{36}_", "", os.path.basename(path))

    file_id = create_file_node(model_name, file_name, "table")
    tables = extract_table_content(path)
    file_stem = re.sub(r"[^\w\-. ]", "_", file_name.rsplit(".", 1)[0])
    table_folder = "data/tables/"

    st.success(f"Extracted {len(tables)} table(s) from {file_name}.....")

    elements = []
    table_records = []
    for table in tables:
        # One csv per table; Excel sheets are suffixed with the sheet name.
        if table["sheet_name"] is None:
            table_name = file_name
            csv_name = f"{uuid.uuid4()}_{file_stem}.csv"
        else:
            table_name = str(table["sheet_name"])
            sheet_name = re.sub(r"[^\w\-. ]", "_", table_name)
            csv_name = f"{uuid.uuid4()}_{file_stem}__{sheet_name}.csv"

        csv_path = f"{table_folder}{csv_name}"
        table["dataframe"].to_csv(csv_path, index=False)

        dataframe = table["dataframe"]
        column_names = [str(column) for column in dataframe.columns]
        column_types = [f"{column}: {dataframe.dtypes[column]}" for column in dataframe.columns]

        elements.append({
            "kind": "table",
            "table_name": table_name,
            "csv_path": csv_path,
            "column_names": column_names,
            "column_types": column_types,
            "row_count": len(dataframe),
        })

        table_records.append({
            "table_name": table_name,
            "csv_path": csv_path,
            "top_5_rows": dataframe.head(5).to_dict(orient="records"),
            "column_names": column_names,
            "column_types": column_types,
            "row_count": len(dataframe),
        })

    # Table nodes carry no content to embed; the ids come back in sheet order.
    stored = store_file_elements(file_id, elements)
    for record, table_id in zip(table_records, stored["table_ids"]):
        record["table_id"] = table_id

    if table_records:
        st.success("Tables stored in Neo4j.....")
    else:
        st.warning("No tables found in file.....")

    if return_ids:
        return {
            "result": "Storing table operation completed.....",
            "file_name": file_name,
            "file_id": file_id,
            "tables": table_records,
        }

    return "Storing table operation completed....."


def store_txt(path: str, model_name: str = str(uuid.uuid4()), chunking_method: str = "text_structure_based", file_name: str = None, return_ids: bool = False):
    """
    Arg:
        path: path to the txt file.
        chunking_method: What type of chunking do you want to apply on your txt text content. Possible inputs are ["text_structure_based", "markdown_based", "semantic_chunker"]
        file_name: original uploaded file name; derived from the path when omitted (uploads land in temp_<uuid>_<name> files).
        return_ids: with True, return a dict with the file's Neo4j node ids (file id, chunk ids) instead of the status string.

    """
    if file_name is None:
        file_name = re.sub(r"^temp_[0-9a-fA-F-]{36}_", "", os.path.basename(path))

    file_id = create_file_node(model_name, file_name, "txt")

    with open(path, "r", encoding="utf-8") as f:
        text_content = f.read()

    documents = chunked_documents(text_content, chunking_method=chunking_method)

    if documents is not None:
        st.success("Chunking of TXT completed.....")
    else:
        st.warning("Chunking issue with TXT.....!!")
        return

    chunk_ids = store_file_elements(file_id, [{"kind": "chunk", "content": document} for document in documents])["chunk_ids"]
    if chunk_ids:
        st.success("TXT data stored in Neo4j.....")
    else:
        st.warning("TXT data was not stored in Neo4j.....!!")

    if return_ids:
        return {
            "result": "Storing TXT operation completed.....",
            "file_name": file_name,
            "file_id": file_id,
            "chunk_ids": chunk_ids,
        }

    return "Storing TXT operation completed....."


if __name__ == "__main__":
    # chunk_method = ["text_structure_based", "markdown_based", "semantic_chunker"]
    # pdf_content = store_pdf(path="data/temp/bkn_duronto.pdf", chunking_method=chunk_method[1])

    # for content in pdf_content:
    #     print(content, end="\n\n")
    # print(pdf_content)


    result = store_image(path="data/images/ebe60544-cb00-4010-a0eb-1c082b53389a_resume_rohit_sharma.jpg", model_name="test_model", return_ids=True)
    print(type(result))
    print(result)
