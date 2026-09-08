import os
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
BASE_URL = os.getenv("BASE_URL")


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


if __name__ == "__main__":
    documents = [
        "Rohit Sharma codes a lot",
        "Rohit Sharma likes to go to gym",
        "Rohit Sharma is looking for growth"
    ]
    embeddings = txt_to_embeddings(documents)
    print(len(embeddings))
    for i in range(0, len(embeddings)):
        print(len(embeddings[i]))