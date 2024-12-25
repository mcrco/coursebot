from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.schema import Document
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain import hub
from dotenv import load_dotenv
import json

if not load_dotenv():
    print("unable to get env vars")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1.0,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = InMemoryVectorStore(embeddings)

with open("../json/cms.json", "r") as f:
    data: dict[str, dict[str, str | list[str] | bool]] = json.load(f)

documents: list[Document] = []
for id, course in data.items():
    chunk = "\n".join([f"{k}: {str(v)}" for k, v in course.items()])
    documents.append(Document(page_content=chunk, metadata={"id": id}))

_ = vector_store.add_documents(documents=documents)

prompt = hub.pull("rlm/rag-prompt")


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=8)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant that helps with course selection at Caltech. "
        "Use the following pieces of retrieved context about Caltech courses to "
        "answer the question. If you don't know the answer, say that you  "
        "don't know.\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human, system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

input_message = "Tell me all the classes I have to take before taking CS 155. This includes any prerequisites of prerequisites."
final_state = graph.invoke({"messages": [{"role": "user", "content": input_message}]})
print(final_state["messages"][-1].content)
