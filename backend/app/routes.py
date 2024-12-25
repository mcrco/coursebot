from flask import Blueprint, request, jsonify
from llm.rag import CourseRAG

api = Blueprint("api", __name__)
rag = CourseRAG()


@api.route("/query", methods=["POST"])
def query():
    input_data = request.json
    if not input_data or "message" not in input_data:
        return jsonify(
            {"error": "Invalid input, expected JSON with 'message' field"}
        ), 400

    input_message = input_data["message"]
    response = rag.answer(input_message)
    return jsonify({"response": response})
