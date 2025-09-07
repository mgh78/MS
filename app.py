import os
from flask import Flask, render_template, request, jsonify

# Reuse the existing RAG pipeline from agent1.py
from agent1 import answer_question


def create_app() -> Flask:
    app = Flask(__name__)

    default_collection = os.getenv("RAG_COLLECTION", "collection_ms")
    default_top_k = int(os.getenv("RAG_TOP_K", "3"))
    default_threshold = float(os.getenv("RAG_THRESHOLD", "0.6"))
    default_llm = os.getenv("RAG_LLM", "llama3")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/ask")
    def ask_api():
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

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)


