import os
from flask import Flask, render_template, request, jsonify

# Reuse the existing RAG pipeline from agent1.py
from agent1 import answer_question
from agent_local import invoke_agent_with_question


def create_app() -> Flask:
    app = Flask(__name__)

    default_collection = os.getenv("RAG_COLLECTION", "collection_ms")
    default_top_k = int(os.getenv("RAG_TOP_K", "3"))
    default_threshold = float(os.getenv("RAG_THRESHOLD", "0.6"))
    default_llm = os.getenv("RAG_LLM", "llama3")

    @app.get("/")
    def index():
        return render_template("index.html")

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

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)

