import argparse
from typing import List
from functools import lru_cache

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")


@lru_cache(maxsize=1)
def get_embeddings_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


@lru_cache(maxsize=256)
def embed_text_cached(text: str) -> List[float]:
    return get_embeddings_model().embed_query(text)


@lru_cache(maxsize=2)
def get_chat_model(name: str) -> ChatOllama:
    return ChatOllama(model=name)

# Use wellness.txt which we already extracted from the PDF
file_path = "/Users/mahdi/Desktop/RAG_MS/wellness.txt"
loader = TextLoader(file_path, encoding="utf-8")


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def build_collection(collection_name: str = "collection_ms") -> None:
    docs = loader.load()
    chunks = split_documents(docs)
    # Infer embedding dimension dynamically
    sample_vector = embed_text_cached(chunks[0].page_content if chunks else "sample")
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
        vec = embed_text_cached(d.page_content)
        points.append(
            PointStruct(id=i, vector=vec, payload={"text": d.page_content})
        )

    client.upsert(collection_name=collection_name, points=points)


def search(query: str, collection_name: str = "collection_ms", top_k: int = 3):
    qvec = embed_text_cached(query)
    results = client.query_points(
        collection_name=collection_name,
        query=qvec,
        limit=top_k,
        with_vectors=False,
        with_payload=True,
    )
    # Normalize return across qdrant-client versions
    points = getattr(results, "points", results)
    if isinstance(points, tuple):
        points = points[0]

    def safe_get(obj, name, default=None):
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    output = []
    for item in points or []:
        payload = safe_get(item, "payload", {}) or {}
        score = safe_get(item, "score", 0.0) or 0.0
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        try:
            score = float(score)
        except Exception:
            score = 0.0
        output.append((text, score))
    return output


def answer_question(
    question: str,
    collection_name: str = "collection_ms",
    top_k: int = 3,
    threshold: float = 0.3,
    model: str = "llama3",
) -> str:
    scored = search(question, collection_name=collection_name, top_k=top_k)
    if not scored:
        return "I don't know."

    best_score = max(score for _, score in scored)
    if best_score < threshold:
        return "I don't know."

    context = "\n\n".join(chunk for chunk, _ in scored)

    prompt = (
        "You are a helpful assistant. Use ONLY the context to answer.\n"
        "If the answer is not clearly contained in the context, say: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    chat = get_chat_model(model)
    response = chat.invoke(prompt)
    text = getattr(response, "content", "").strip()
    if not text:
        return "I don't know."
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG over wellness.txt with Qdrant + Ollama embeddings")
    parser.add_argument("--build", action="store_true", help="Build/rebuild the vector collection")
    parser.add_argument("--query", type=str, default="", help="Run a semantic search query")
    parser.add_argument("--ask", type=str, default="", help="Ask a question; answers only if supported by context, else 'I don't know'")
    parser.add_argument("--top_k", type=int, default=3, help="Top K results to return")
    parser.add_argument("--collection", type=str, default="collection_ms", help="Qdrant collection name")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for answering (higher = more precise, more 'I don't know')",
    )
    parser.add_argument(
        "--show_scores",
        action="store_true",
        help="When used with --ask, prints the retrieved chunk scores to help tune threshold",
    )
    parser.add_argument("--llm", type=str, default="llama3", help="Ollama chat model for answering")
    args = parser.parse_args()

    if args.build:
        build_collection(collection_name=args.collection)
        print("Collection built.")

    if args.query:
        results = search(args.query, collection_name=args.collection, top_k=args.top_k)
        for text, score in results:
            print(f"[score={score:.3f}]\n{text}\n")

    if args.ask:
        ans = answer_question(
            args.ask,
            collection_name=args.collection,
            top_k=args.top_k,
            threshold=args.threshold,
            model=args.llm,
        )
        print(ans)


if __name__ == "__main__":
    main()
