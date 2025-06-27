from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import (
    SystemMessage,
    BaseMessage,
    ToolMessage,
    AIMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from dotenv import load_dotenv
import os
from llm.search import hybrid_search, course_catalog_search
from typing import Annotated, Sequence, TypedDict
from functools import partial
import operator
import json


MODEL_CODE = "qwen/qwen3-30b-a3b"
TEMPERATURE = 0.69
SYSTEM_MESSAGE_CONTENT = open("llm/react_prompt.txt", "r").read()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class CourseRAG:
    def __init__(
        self,
        model_code=MODEL_CODE,
        embedding_model="models/text-embedding-004",
        sparse_embedding_model="Qdrant/bm25",
    ):
        if not load_dotenv():
            print("Unable to get environment variables via pydotenv.")

        self.llm = ChatOpenAI(
            model=model_code,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=TEMPERATURE,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_embedding_model)
        self.vector_store = QdrantVectorStore.from_existing_collection(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            collection_name="coursebot_hybrid",
            content_payload_key="text",
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            vector_name="dense_vector",
            sparse_vector_name="sparse_vector",
            retrieval_mode=RetrievalMode.SPARSE,
        )

        self.tools = [hybrid_search, course_catalog_search]
        self.tool_executor = ToolExecutor(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.build_graph()

    def build_graph(self):
        graph_builder = StateGraph(AgentState)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_CONTENT),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.agent = prompt | self.llm_with_tools

        graph_builder.add_node("agent", self.run_agent)
        graph_builder.add_node("action", self.execute_tools)
        graph_builder.set_entry_point("agent")
        graph_builder.add_conditional_edges(
            "agent", self.should_continue, {"action": "action", "end": END}
        )
        graph_builder.add_edge("action", "agent")

        self.graph = graph_builder.compile()

    def run_agent(self, state):
        """
        Think about what to do
        """
        messages = state["messages"]
        response = self.agent.invoke({"messages": messages})
        return {"messages": [response]}

    def execute_tools(self, state):
        """
        Execute tools
        """
        messages = state["messages"]
        last_message = messages[-1]

        tool_invocations = []
        for tool_call in last_message.tool_calls:
            args = tool_call["args"].copy()
            args["vector_store"] = self.vector_store
            tool_invocations.append(
                ToolInvocation(tool=tool_call["name"], tool_input=args)
            )

        responses = self.tool_executor.batch(tool_invocations, return_exceptions=True)
        tool_messages = [
            ToolMessage(content=str(res), tool_call_id=tool_call["id"])
            for res, tool_call in zip(responses, last_message.tool_calls)
        ]
        
        return {"messages": tool_messages}

    def should_continue(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        return "action"

    def stream_complete(self, messages):
        state = {"messages": messages}
        for chunk in self.graph.stream(state):
            if "agent" in chunk:
                last_message = chunk["agent"]["messages"][-1]
                if last_message.tool_calls:
                    # The new plan is the set of tool calls
                    new_plan = {
                        "steps": [
                            {"tool_name": tc["name"], "args": tc["args"]}
                            for tc in last_message.tool_calls
                        ]
                    }
                    yield f"data: {json.dumps({'type': 'plan', 'data': new_plan})}\n\n"
                    yield f"data: {json.dumps({'type': 'tools_start'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'final_answer', 'data': last_message.content})}\n\n"
            if "action" in chunk:
                yield f"data: {json.dumps({'type': 'tools_end', 'data': [msg.content for msg in chunk['action']['messages']]})}\n\n"
