from flask import Blueprint, request, jsonify, Response
from langchain_core.messages import AIMessage, HumanMessage
from llm.rag import CourseRAG

api = Blueprint("api", __name__)
rag = CourseRAG()


@api.route("/query", methods=["POST"])
def query():
    input_data = request.json
    if not input_data or "messages" not in input_data:
        return jsonify(
            {"error": "Invalid input, expected JSON with 'message' field"}
        ), 400

    input_messages = input_data["messages"]
    response_generator = rag.stream_complete(input_messages)

    def generate():
        for message in response_generator:
            yield message.content

    return Response(generate(), mimetype="text/event-stream")

