import os
from flask import Flask, render_template, request, jsonify, Response

# Reuse the existing RAG pipeline from agent1.py
from agent1 import answer_question, get_embeddings_model, get_chat_model, search
from agent_local import invoke_agent_with_question
from agent2 import answer_question as pinecone_answer_question, get_embeddings_model as pinecone_get_embeddings_model, get_chat_model as pinecone_get_chat_model


def create_app() -> Flask:
    app = Flask(__name__)

    default_collection = os.getenv("RAG_COLLECTION", "collection_ms")
    default_top_k = int(os.getenv("RAG_TOP_K", "3"))
    default_threshold = float(os.getenv("RAG_THRESHOLD", "0.6"))
    default_llm = os.getenv("RAG_LLM", "qwen2.5:7b-instruct-q4_0")

    # Prewarm local models so the first UI request is fast
    try:
        get_embeddings_model()
        get_chat_model(default_llm)
        # Also prewarm Pinecone models
        pinecone_get_embeddings_model()
        pinecone_get_chat_model(default_llm)
    except Exception:
        # Prewarm failures should not crash the server
        pass

    @app.get("/")
    def index():
        return render_template("index.html")
    
    @app.get("/pinecone")
    def pinecone_index():
        return render_template("pinecone.html")

    # Pure RAG endpoint
    @app.post("/api/rag")
    def ask_rag():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400

        # Allow optional overrides per request
        collection = data.get("collection") or default_collection
        top_k = int(data.get("top_k") or default_top_k)
        threshold = float(data.get("threshold") or default_threshold)
        llm = data.get("llm") or default_llm

        answer = answer_question(
            question,
            collection_name=collection,
            top_k=top_k,
            threshold=threshold,
            model=llm,
        )
        return jsonify({"answer": answer})

    # Agent endpoint with web-search fallback
    @app.post("/api/agent")
    def ask_agent():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        # Prefer the smart answer tool's logic directly for reliability
        try:
            from agent_local import smart_answer  # Tool object
            # Call the tool directly to ensure we always run RAG→web fallback
            ans = smart_answer.invoke(question)
        except Exception:
            # As a fallback, try the agent wrapper
            ans = invoke_agent_with_question(question)
        # Ensure a non-empty string so UI shows something meaningful
        answer = (ans or "I don't know.").strip()
        return jsonify({"answer": answer})

    # Web-only endpoint (triggered when user clicks "Search the web")
    @app.post("/api/web")
    def ask_web():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        try:
            from agent_local import web_answer
            ans = web_answer(question)
        except Exception:
            ans = "I don't know."
        return jsonify({"answer": (ans or "I don't know.").strip()})

    # Streaming RAG (GET with ?q=...)
    @app.get("/api/rag_stream")
    def rag_stream():
        question = (request.args.get("q") or "").strip()
        if not question:
            return Response("question is required", mimetype="text/plain")

        collection = request.args.get("collection") or default_collection
        top_k = int(request.args.get("top_k") or default_top_k)
        threshold = float(request.args.get("threshold") or default_threshold)
        llm = request.args.get("llm") or default_llm

        def generate():
            try:
                scored = search(question, collection_name=collection, top_k=top_k)
                if not scored:
                    yield "I don't know."
                    return
                best_score = max(score for _, score in scored)
                if best_score < threshold:
                    yield "I don't know."
                    return
                context = "\n\n".join(chunk for chunk, _ in scored)
                prompt = (
                    "You are a helpful assistant. Use ONLY the context to answer.\n"
                    "If the answer is not clearly contained in the context, say: I don't know.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
                model = get_chat_model(llm)
                # Stream tokens if available
                try:
                    for chunk in model.stream(prompt):  # type: ignore[attr-defined]
                        text = getattr(chunk, "content", None)
                        if text:
                            yield text
                except Exception:
                    # Fallback to single invoke
                    resp = model.invoke(prompt)
                    text = getattr(resp, "content", "")
                    yield text
            except Exception:
                yield "I don't know."

        return Response(generate(), mimetype="text/plain")

        
    # Streaming web (GET with ?q=...)
    @app.get("/api/web_stream")
    def web_stream():
        question = (request.args.get("q") or "").strip()
        if not question:
            return Response("question is required", mimetype="text/plain")
        try:
            from agent_local import TavilySearch, ChatOllama  # type: ignore
        except Exception:
            # Safety import fallback
            from langchain_tavily import TavilySearch  # type: ignore
            from langchain_ollama import ChatOllama  # type: ignore

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return Response("I don't know. (Enable web search by setting TAVILY_API_KEY)", mimetype="text/plain")

        def generate():
            try:
                tav = TavilySearch(max_results=3, api_key=api_key)
                results = tav.invoke(question) or []
                items = results.get("results", results) if isinstance(results, dict) else results
                snippets = []
                for item in items or []:
                    text = item.get("content") if isinstance(item, dict) else (item or "")
                    if not text:
                        text = item.get("snippet", "") if isinstance(item, dict) else ""
                    text = str(text).strip()
                    if len(text) > 10:
                        snippets.append(text)
                if not snippets:
                    yield "I don't know."
                    return
                context = "\n\n".join(snippets[:3])
                prompt = (
                    "You are a helpful assistant. Use ONLY the web snippets to answer briefly.\n"
                    "If the answer is not clearly contained in the snippets, say: I don't know.\n\n"
                    f"Snippets:\n{context}\n\n"
                    f"Question: {question}\nAnswer:"
                )
                model = ChatOllama(model=default_llm)
                try:
                    for chunk in model.stream(prompt):  # type: ignore[attr-defined]
                        text = getattr(chunk, "content", None)
                        if text:
                            yield text
                except Exception:
                    resp = model.invoke(prompt)
                    yield getattr(resp, "content", "")
            except Exception:
                yield "I don't know."

        return Response(generate(), mimetype="text/plain")

    # Pinecone RAG endpoint
    @app.post("/api/pinecone")
    def ask_pinecone():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400

        # Allow optional overrides per request
        top_k = int(data.get("top_k") or 2)
        threshold = float(data.get("threshold") or 0.6)
        llm = data.get("llm") or default_llm
        namespace = data.get("namespace") or "wellness-v1"

        answer = pinecone_answer_question(
            question,
            top_k=top_k,
            threshold=threshold,
            model=llm,
            namespace=namespace,
        )
        return jsonify({"answer": answer})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)

