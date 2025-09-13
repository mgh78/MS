import os
from typing import List, Tuple
from functools import lru_cache

from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings, ChatOllama


# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_72p94R_2pTCU6QAfU9ySsJeqVLEchvkkjmYMsLP9JLxgpAQrq5o2hdCWLGHPgdgbEzHvjS")
INDEX_NAME = os.getenv("PINECONE_INDEX", "ms")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "wellness-v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.getenv("RAG_LLM", "qwen2.5:7b-instruct-q4_0")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


@lru_cache(maxsize=1)
def get_embeddings_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


@lru_cache(maxsize=2)
def get_chat_model(name: str) -> ChatOllama:
    return ChatOllama(model=name, temperature=0.3, num_predict=384)


def search_scored(query: str, top_k: int = 2, namespace: str = NAMESPACE) -> List[Tuple[str, float]]:
    """Search Pinecone and return scored results."""
    qvec = get_embeddings_model().embed_query(query)
    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        namespace=namespace,
    )
    return [
        ((m.get("metadata") or {}).get("text", ""), m.get("score", 0.0))
        for m in res.get("matches", [])
    ]


def answer_question(
    question: str,
    top_k: int = 2,
    threshold: float = 0.6,
    model: str = CHAT_MODEL,
    namespace: str = NAMESPACE,
) -> str:
    """Answer question using Pinecone RAG."""
    scored = search_scored(question, top_k=top_k, namespace=namespace)
    if not scored:
        return "I don't know."
    
    best_score = max(score for _, score in scored)
    if best_score < threshold:
        return "I don't know."
    
    context = "\n\n".join(chunk for chunk, _ in scored)
    if len(context) > 4000:
        context = context[:4000]
    
    prompt = (
        "You are a helpful assistant. Answer in the same language as the question "
        "(prefer Persian if the question is Persian). "
        "Use ONLY the context to answer. If the answer is not clearly contained in the context, say: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    
    chat = get_chat_model(model)
    response = chat.invoke(prompt)
    text = getattr(response, "content", "").strip()
    return text or "I don't know."


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Pinecone RAG")
    parser.add_argument("--ask", type=str, default="", help="Ask a question")
    parser.add_argument("--top_k", type=int, default=2, help="Top K results")
    parser.add_argument("--threshold", type=float, default=0.6, help="Answer threshold")
    parser.add_argument("--llm", type=str, default=CHAT_MODEL, help="Chat model")
    args = parser.parse_args()
    
    if args.ask:
        print(answer_question(args.ask, top_k=args.top_k, threshold=args.threshold, model=args.llm))


if __name__ == "__main__":
    main()






