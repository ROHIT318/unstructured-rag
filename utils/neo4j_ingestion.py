import os
import uuid
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph

load_dotenv()

# Embeddings generator credentials
EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
BASE_URL = os.getenv("BASE_URL")

# Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

# One shared dimension setting governs all text embeddings — chunk embeddings
# and the image nodes' semantic / OCR text embeddings alike (§"Embedding
# dimension" of the image OCR spec). Changing it is a one-place change.
TEXT_EMBEDDING_DIMENSION = 256

# Node labels and their id properties — delete_nodes_by_ids goes through this
# whitelist, never through caller-supplied Cypher fragments.
_NODE_KINDS = {
    "model": ("Model", "model_id"),
    "file": ("File", "file_id"),
    "chunk": ("Chunk", "chunk_id"),
    "image": ("Image", "image_id"),
    "table": ("Table", "table_id"),
}


def txt_to_embeddings(documents: List[str], embeddings_dimension: int = TEXT_EMBEDDING_DIMENSION) -> List[float]:
    """
    Turns a list of string into embeddings and return it.

    Arg:
        documents: List of strings that is to be converted into embeddings.
        embedding_dimension: An integer defining the dimension of output embeddings.

    Returns:
        embedings: List of embeddings of dimension len(documents) and each embedding element having length equals to embeddings_dimension.

    """
    embedding_model_kwargs = {
        "api_key": EMBEDDING_API_KEY,
        "model": EMBEDDING_MODEL,
        "output_dimensionality": embeddings_dimension
    }
    if BASE_URL:
        embedding_model_kwargs["base_url"] = BASE_URL
    embedding_model = GoogleGenerativeAIEmbeddings(**embedding_model_kwargs)

    embeddings = embedding_model.embed_documents(documents)

    return embeddings


def setup_schema() -> None:
    """
    Uniqueness constraints on every node label's id property (and on
    Model.model_name, the key models are looked up by), plus the three vector
    indexes: the Chunk embedding index and the two Image text-embedding indexes
    (semantic meaning, OCR). All share TEXT_EMBEDDING_DIMENSION, so all are
    created here at fixed dimension — no lazy creation needed.
    """
    for label, id_property in _NODE_KINDS.values():
        graph.query(
            f"CREATE CONSTRAINT {label.lower()}_{id_property}_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{id_property} IS UNIQUE"
        )
    graph.query(
        f"""CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {TEXT_EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}}}"""
    )
    graph.query(
        f"""CREATE VECTOR INDEX image_semantic_embedding_index IF NOT EXISTS
            FOR (i:Image) ON (i.semantic_embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {TEXT_EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}}}"""
    )
    graph.query(
        f"""CREATE VECTOR INDEX image_ocr_embedding_index IF NOT EXISTS
            FOR (i:Image) ON (i.ocr_embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {TEXT_EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}}}"""
    )


def ensure_model_node(model_name: str, model_description: str = "") -> str:
    """
    Get or create the model's Model node and return its model_id. An empty
    description never erases a recorded one (add-files passes none).
    """
    result = graph.query(
        """
        MERGE (m:Model {model_name: $model_name})
        ON CREATE SET m.model_id = $model_id
        SET m.model_description = CASE
            WHEN $model_description IS NULL OR $model_description = ''
            THEN coalesce(m.model_description, '')
            ELSE $model_description
        END
        RETURN m.model_id AS model_id
        """,
        params={
            "model_name": model_name,
            "model_id": str(uuid.uuid4()),
            "model_description": model_description or "",
        },
    )
    return result[0]["model_id"]


def update_model_node_description(model_name: str, description: str) -> None:
    """Set the Model node's description (kept in sync with the record header)."""
    graph.query(
        "MATCH (m:Model {model_name: $model_name}) SET m.model_description = $description",
        params={"model_name": model_name, "description": description},
    )


def create_file_node(model_name: str, file_name: str, file_type: str, file_description: str = "") -> str:
    """
    Create a File node for one upload and link it to the model's Model node
    (creating that node if it does not exist yet). Every upload gets a fresh
    file_id — re-uploads are separate entries, never merged into the old File.
    """
    ensure_model_node(model_name)
    file_id = str(uuid.uuid4())
    graph.query(
        """
        MATCH (m:Model {model_name: $model_name})
        CREATE (f:File {file_id: $file_id, file_name: $file_name, file_type: $file_type, file_description: $file_description})
        CREATE (m)-[:HAS_FILE]->(f)
        """,
        params={
            "model_name": model_name,
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "file_description": file_description,
        },
    )
    return file_id


def store_file_elements(file_id: str, elements: List[dict]) -> dict:
    """
    Create a file's element nodes in the given document order and chain them.

    Elements are dicts, one per chunk / image / table, in the order they appear
    in the source document — page by page for a pdf, slide by slide for a pptx,
    sheet by sheet for a workbook. The kinds:

        {"kind": "chunk", "content": str or langchain Document}
        {"kind": "image", "file_name": str, "file_path": str,
         "semantic_meaning": str, "semantic_embedding": list,
         "ocr_text": str, "ocr_embedding": list}   # ocr_embedding omitted
                                                   # when ocr_text is empty
        {"kind": "table", "table_name": str, "csv_path": str,
         "column_names": list, "column_types": list, "row_count": int}

    Every node carries its element_index (position in the list), and consecutive
    elements are chained with a generic HAS_NEXT relationship regardless of
    kind — chunk -> image, image -> table, whatever the document order is.
    Chunks additionally get a chunk_index counting only the file's chunks, and
    are embedded with Gemini (one call for the whole file).

    Returns {"chunk_ids": [...], "image_ids": [...], "table_ids": [...]}, each
    list in element order.
    """
    chunk_contents = [
        element["content"].page_content if isinstance(element["content"], Document) else str(element["content"])
        for element in elements if element.get("kind") == "chunk"
    ]
    chunk_embeddings = txt_to_embeddings(chunk_contents) if chunk_contents else []

    chunk_rows = []
    image_rows = []
    table_rows = []
    chunk_counter = 0
    for element_index, element in enumerate(elements):
        kind = element.get("kind")
        if kind == "chunk":
            chunk_rows.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": chunk_counter,
                "element_index": element_index,
                "chunk_content": chunk_contents[chunk_counter],
                "embedding": chunk_embeddings[chunk_counter],
            })
            chunk_counter += 1
        elif kind == "image":
            image_row = {
                "image_id": str(uuid.uuid4()),
                "element_index": element_index,
                "file_name": element["file_name"],
                "file_path": element["file_path"],
                "semantic_meaning": element["semantic_meaning"],
                "semantic_embedding": element["semantic_embedding"],
                "ocr_text": element["ocr_text"],
            }
            # No readable text -> no ocr_embedding property at all: the node
            # simply stays out of the OCR vector index.
            if element["ocr_text"]:
                image_row["ocr_embedding"] = element["ocr_embedding"]
            image_rows.append(image_row)
        elif kind == "table":
            table_rows.append({
                "table_id": str(uuid.uuid4()),
                "element_index": element_index,
                "table_name": element["table_name"],
                "csv_path": element["csv_path"],
                "column_names": element["column_names"],
                "column_types": element["column_types"],
                "row_count": element["row_count"],
            })

    if chunk_rows:
        graph.query(
            """
            MATCH (f:File {file_id: $file_id})
            UNWIND $rows AS row
            CREATE (c:Chunk {chunk_id: row.chunk_id, chunk_index: row.chunk_index, element_index: row.element_index, chunk_content: row.chunk_content, embedding: row.embedding})
            CREATE (f)-[:HAS_CHUNK]->(c)
            """,
            params={"file_id": file_id, "rows": chunk_rows},
        )
    if image_rows:
        graph.query(
            """
            MATCH (f:File {file_id: $file_id})
            UNWIND $rows AS row
            CREATE (i:Image {image_id: row.image_id, element_index: row.element_index, file_name: row.file_name, file_path: row.file_path, semantic_meaning: row.semantic_meaning, semantic_embedding: row.semantic_embedding, ocr_text: row.ocr_text})
            CREATE (f)-[:HAS_IMAGE]->(i)
            WITH i, row WHERE row.ocr_embedding IS NOT NULL
            SET i.ocr_embedding = row.ocr_embedding
            """,
            params={"file_id": file_id, "rows": image_rows},
        )
    if table_rows:
        graph.query(
            """
            MATCH (f:File {file_id: $file_id})
            UNWIND $rows AS row
            CREATE (t:Table {table_id: row.table_id, element_index: row.element_index, table_name: row.table_name, csv_path: row.csv_path, column_names: row.column_names, column_types: row.column_types, row_count: row.row_count})
            CREATE (f)-[:HAS_TABLE]->(t)
            """,
            params={"file_id": file_id, "rows": table_rows},
        )

    # Chain consecutive elements by their document position — matched afresh, so
    # the chain never depends on collect() ordering.
    graph.query(
        """
        MATCH (f:File {file_id: $file_id})-[:HAS_CHUNK|HAS_IMAGE|HAS_TABLE]->(e1),
              (f)-[:HAS_CHUNK|HAS_IMAGE|HAS_TABLE]->(e2)
        WHERE e2.element_index = e1.element_index + 1
        CREATE (e1)-[:HAS_NEXT]->(e2)
        """,
        params={"file_id": file_id},
    )

    return {
        "chunk_ids": [row["chunk_id"] for row in chunk_rows],
        "image_ids": [row["image_id"] for row in image_rows],
        "table_ids": [row["table_id"] for row in table_rows],
    }


def get_model_names() -> list:
    """Underlying names of every Model node in the graph."""
    result = graph.query("MATCH (m:Model) RETURN m.model_name AS model_name")
    return sorted({row["model_name"] for row in result if row.get("model_name")})


def model_node_counts(model_name: str) -> dict:
    """Best-effort counts of a model's chunks and images across its files ({} when it has no Model node)."""
    try:
        result = graph.query(
            """
            MATCH (m:Model {model_name: $model_name})-[:HAS_FILE]->(f:File)
            RETURN sum(size([(f)-[:HAS_CHUNK]->() | 1])) AS text,
                   sum(size([(f)-[:HAS_IMAGE]->() | 1])) AS images
            """,
            params={"model_name": model_name},
        )
        if not result:
            return {}
        return {"text": result[0]["text"] or 0, "images": result[0]["images"] or 0}
    except Exception:
        return {}


def get_model_artifact_paths(model_name: str) -> list:
    """Disk paths (data/images/, data/tables/) the model's Neo4j subtree points at."""
    result = graph.query(
        """
        MATCH (m:Model {model_name: $model_name})-[:HAS_FILE]->(f:File)
        OPTIONAL MATCH (f)-[:HAS_IMAGE]->(i:Image)
        OPTIONAL MATCH (f)-[:HAS_TABLE]->(t:Table)
        RETURN collect(DISTINCT i.file_path) AS image_paths, collect(DISTINCT t.csv_path) AS csv_paths
        """,
        params={"model_name": model_name},
    )
    if not result:
        return []
    paths = (result[0].get("image_paths") or []) + (result[0].get("csv_paths") or [])
    return [path for path in paths if path]


def delete_nodes_by_ids(kind: str, ids: list) -> int:
    """
    Delete nodes of one label by their recorded ids — DETACH, so their
    relationships go with them. Scoping to the recorded ids is what keeps a
    re-uploaded same-named file's nodes safe. Returns how many were deleted;
    ids that no longer exist are not errors.
    """
    if not ids:
        return 0
    label, id_property = _NODE_KINDS[kind]
    result = graph.query(
        f"MATCH (n:{label}) WHERE n.{id_property} IN $ids DETACH DELETE n RETURN count(n) AS deleted",
        params={"ids": ids},
    )
    return result[0]["deleted"] if result else 0


def delete_model_subtree(model_name: str) -> int:
    """
    Delete the Model node and its entire subtree — files with their chunks,
    images and tables, and every relationship between them. Children first
    (DETACH also drops the HAS_NEXT chains), then the File nodes, then the
    Model node itself. Missing nodes are not errors. Returns the number of
    deleted nodes (0 when the model has no Model node).
    """
    deleted = 0
    for query in (
        """
        MATCH (m:Model {model_name: $model_name})-[:HAS_FILE]->(f:File)
        MATCH (f)-[:HAS_CHUNK|HAS_IMAGE|HAS_TABLE]->(n)
        DETACH DELETE n
        RETURN count(n) AS deleted
        """,
        """
        MATCH (m:Model {model_name: $model_name})-[:HAS_FILE]->(f:File)
        DETACH DELETE f
        RETURN count(f) AS deleted
        """,
        """
        MATCH (m:Model {model_name: $model_name})
        DETACH DELETE m
        RETURN count(m) AS deleted
        """,
    ):
        result = graph.query(query, params={"model_name": model_name})
        deleted += result[0]["deleted"] if result else 0
    return deleted


setup_schema()


if __name__ == "__main__":
    documents = [
        "Rohit Sharma codes a lot",
        "Rohit Sharma likes to go to gym",
        "Rohit Sharma is looking for growth"
    ]

    model_name = "test_model"
    ensure_model_node(model_name, model_description="Test model description")
    file_id = create_file_node(model_name, file_name="text.txt", file_type="txt")

    elements = [{"kind": "chunk", "content": document} for document in documents]
    stored = store_file_elements(file_id, elements)

    print(file_id)
    print(stored)
