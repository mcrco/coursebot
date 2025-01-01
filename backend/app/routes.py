from flask import Blueprint, request, jsonify
from langchain_core.messages import AIMessage, HumanMessage
from llm.rag import CourseRAG


api = Blueprint("api", __name__)
rag = CourseRAG()


def msg2json(message: HumanMessage | AIMessage, idx):
    if isinstance(message, HumanMessage):
        return {"id": str(idx), "role": "user", "content": message.content}
    else:
        return {"id": str(idx), "role": "assistant", "content": message.content}


@api.route("/query", methods=["POST"])
def query():
    input_data = request.json
    if not input_data or "messages" not in input_data:
        return jsonify(
            {"error": "Invalid input, expected JSON with 'message' field"}
        ), 400

    input_messages = input_data["messages"]
    output_message = rag.complete(input_messages)
    return jsonify({"response": msg2json(output_message, len(input_messages) + 1)})
