import os
import uuid
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph

load_dotenv()

# Embeddings generator credentials
EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
BASE_URL = os.getenv("BASE_URL")

# Neoj Credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)


def txt_to_embeddings(documents: List[str], embeddings_dimension: int = 256) -> List[float]:
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

def create_model_node(model_id: str, model_name: str, model_description: str) -> str:
    model_node_creation_query = """
    MERGE (d:Model {model_id: $model_id, model_name: $model_name, model_description: $model_description})
    RETURN d
    """
    graph.query(model_node_creation_query, params={"model_id": model_id, "model_name": model_name, "model_description": model_description})
    graph.refresh_schema()

    return f"Model Node Creation Successfull for {model_name}....."


def create_file_node(file_id: str, file_name: str, file_type: str, file_description: str) -> str:
    file_node_creation_query = """
    MERGE (d:File {file_id: $file_id, file_name: $file_name, file_type: $file_type, file_description: $file_description})
    RETURN d
    """
    graph.query(file_node_creation_query, params={"file_id": file_id, "file_name": file_name, "file_type": file_type, "file_description": file_description})

    return f"File Node Creation Successfull for {file_name}....."


def create_chunk_node(chunk_id: str, chunk_index: str, chunk_content: str, embedding: List[float]) -> str:
    chunk_node_creation_query = """
    MERGE (d:Chunk {chunk_id: $chunk_id, chunk_index: $chunk_index, chunk_content: $chunk_content, embedding: $embedding})
    RETURN d
    """
    graph.query(chunk_node_creation_query, params={"chunk_id": chunk_id, "chunk_index": chunk_index, "chunk_content": chunk_content, "embedding": embedding})

    return f"Chunk Node Creation Successfull for {chunk_index}....."


def link_file_to_model(model_id: str, file_id: str) -> str:
    file_to_model_relationship_query = """
    MATCH (m:Model {model_id: $model_id}), (f:File {file_id: $file_id})
    MERGE (m)-[:HAS_FILE]->(f)
    """
    graph.query(file_to_model_relationship_query, params={"model_id": model_id, "file_id": file_id})

    return f"Relationship created between Model: {model_id} and File: {file_id}....."


def link_chunk_to_file(file_id: str, chunk_id: str) -> str:
    file_to_model_relationship_query = """
    MATCH (c:Chunk {chunk_id: $chunk_id}), (f:File {file_id: $file_id})
    MERGE (f)-[:HAS_CHUNK]->(c)
    """
    graph.query(file_to_model_relationship_query, params={"chunk_id": chunk_id, "file_id": file_id})

    return f"Relationship created between File: {file_id} and Chunk: {chunk_id}....."


if __name__ == "__main__":
    documents = [
        "Rohit Sharma codes a lot",
        "Rohit Sharma likes to go to gym",
        "Rohit Sharma is looking for growth"
    ]
    embeddings = txt_to_embeddings(documents)
    # print(len(embeddings))
    # for i in range(0, len(embeddings)):
    #     print(len(embeddings[i]))

    model_name = "test"
    model_id = model_name + "_" + str(uuid.uuid7())
    file_name = "text.txt"
    file_id = file_name.split('.')[0] + "_" + str(uuid.uuid7())


    create_model_node(model_id = model_id, model_name = "test_model_node", model_description = "Test model description")

    create_file_node(file_id = file_id, file_name = "test_file_node", file_type = ".txt", file_description = "Some file description here")
    link_file_to_model(model_id = model_id, file_id = file_id)

    for i in range(0, len(documents)):
        chunk_id = model_name + file_name.split('.')[0] + "_chunk_" + str(uuid.uuid7())
        create_chunk_node(chunk_id = chunk_id, chunk_index = f"chunk_{i}", chunk_content = documents[i], embedding = embeddings[i])
        link_chunk_to_file(file_id = file_id, chunk_id = chunk_id)

    