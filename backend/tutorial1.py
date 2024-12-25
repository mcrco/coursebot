from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.schema import Document
from langgraph.graph import START, StateGraph
from langchain import hub
from dotenv import load_dotenv
import json

if not load_dotenv():
    print("unable to get env vars")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=1.0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    # other params...
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


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    print(retrieved_docs)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is the link for the CS 156 ab course?"})
print(response["answer"])
