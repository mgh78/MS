from functools import lru_cache
from typing import Dict, List, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from agent1 import answer_question


@tool
def smart_answer(question: str) -> str:
    """Answer by first using local RAG; if unavailable, fall back to web search."""
    print(f"🔍 Smart answer called with: {question}")
    
    try:
        rag = answer_question(
            question=question,
            collection_name="collection_ms",
            top_k=3,
            threshold=0.6,
            model="llama3",
        ).strip()
    except Exception as e:
        print(f"❌ RAG pipeline failed: {e}")
        rag = ""
    
    print(f"📚 RAG result: {rag}")

    if rag and not rag.lower().startswith("i don't know"):
        print("✅ Using RAG answer")
        return rag

    print("🌐 RAG failed, trying web search...")
    # Fallback: Web search via Tavily
    tavily = TavilySearch(max_results=3)
    try:
        results = tavily.invoke(question) or []
        print(f"🔍 Web search results: {len(results)} items")
    except Exception as e:
        print(f"❌ Web search failed: {e}")
        # Provide a helpful hint if API key is missing
        import os
        if not os.getenv("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"):
            return "I don't know. (Enable web search by setting TAVILY_API_KEY)"
        results = []

    snippets: List[str] = []
    # Handle different result formats from Tavily
    if isinstance(results, dict) and "results" in results:
        # Tavily returns {"results": [...]}
        items = results["results"]
    elif isinstance(results, list):
        # Direct list of results
        items = results
    else:
        items = []
    
    for item in items:
        if isinstance(item, dict):
            text = item.get("content") or item.get("snippet") or ""
        elif isinstance(item, str):
            text = item
        else:
            text = str(item)
        
        if text and len(text.strip()) > 10:  # Only add substantial content
            snippets.append(text.strip())

    if not snippets:
        print("❌ No web snippets found")
        return "I don't know."

    context = "\n\n".join(snippets[:3])
    prompt = (
        "You are a helpful assistant. Use ONLY the web snippets to answer briefly.\n"
        "If the answer is not clearly contained in the snippets, say: I don't know.\n\n"
        f"Snippets:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    chat = ChatOllama(model="llama3")
    try:
        resp = chat.invoke(prompt)
        text = getattr(resp, "content", "").strip()
        print(f"✅ Web-based answer: {text}")
        return text or "I don't know."
    except Exception as e:
        print(f"❌ LLM failed: {e}")
        return "I don't know."


@lru_cache(maxsize=1)
def get_agent():
    memory = MemorySaver()
    model = ChatOllama(model="llama3")
    tools = [smart_answer]
    agent = create_react_agent(model, tools, checkpointer=memory)
    return agent


def invoke_agent_with_question(question: str) -> str:
    print(f"🤖 Agent invoked with: {question}")
    agent = get_agent()
    input_message = {"role": "user", "content": question}
    result = agent.invoke({"messages": [input_message]})
    print(f"🤖 Agent result: {result}")
    
    # result is typically a dict with "messages" list; take last content
    messages: List[Dict[str, Any]] = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        print("❌ No messages in result")
        return "I don't know."
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else ""
    print(f"🤖 Final answer: {content}")
    return content or "I don't know."
