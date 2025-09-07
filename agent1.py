import argparse
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

# Use wellness.txt which we already extracted from the PDF
file_path = "/Users/mahdi/Desktop/RAG_MS/wellness.txt"
loader = TextLoader(file_path, encoding="utf-8")


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def build_collection(collection_name: str = "collection_ms") -> None:
    docs = loader.load()
    chunks = split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Infer embedding dimension dynamically
    sample_vector = embeddings.embed_query(chunks[0].page_content if chunks else "sample")
    dim = len(sample_vector)

    # Create or recreate collection with correct vector size
    try:
        client.get_collection(collection_name=collection_name)
        # If exists, ensure dimensions match by recreating
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    except Exception:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points: List[PointStruct] = []
    for i, d in enumerate(chunks):
        vec = embeddings.embed_query(d.page_content)
        points.append(
            PointStruct(id=i, vector=vec, payload={"text": d.page_content})
        )

    client.upsert(collection_name=collection_name, points=points)


def search(query: str, collection_name: str = "collection_ms", top_k: int = 5) -> List[str]:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    qvec = embeddings.embed_query(query)
    results = client.search(
        collection_name=collection_name,
        query_vector=qvec,
        limit=top_k,
    )
    return [r.payload.get("text", "") for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG over wellness.txt with Qdrant + Ollama embeddings")
    parser.add_argument("--build", action="store_true", help="Build/rebuild the vector collection")
    parser.add_argument("--query", type=str, default="", help="Run a semantic search query")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results to return")
    parser.add_argument("--collection", type=str, default="collection_ms", help="Qdrant collection name")
    args = parser.parse_args()

    if args.build:
        build_collection(collection_name=args.collection)
        print("Collection built.")

    if args.query:
        hits = search(args.query, collection_name=args.collection, top_k=args.top_k)
        print("\n".join(hits))


if __name__ == "__main__":
    main()

