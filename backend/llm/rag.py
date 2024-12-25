from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone
from dotenv import load_dotenv
import os


class CourseRAG:
    def __init__(
        self, model_code="gpt-4o-mini", embedding_model="text-embedding-3-small"
    ):
        if not load_dotenv():
            print("Unable to get environment variables via pydotenv.")

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index("caltech-courses")

        self.llm = ChatOpenAI(model=model_code, temperature=1.0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = PineconeVectorStore(
            index=self.index, embedding=self.embeddings
        )

        self.build_graph()

    def build_graph(self):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=8)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        self.llm_with_tools = self.llm.bind_tools([retrieve])

        def query_or_respond(state: MessagesState):
            response = self.llm_with_tools.invoke(state["messages"])
            return {"messages": response}

        def generate(state: MessagesState):
            recent_tool_messages = [
                message
                for message in reversed(state["messages"])
                if message.type == "tool"
            ][::-1]

            docs_content = "\n\n".join(
                str(doc.content) if not isinstance(doc.content, str) else doc.content
                for doc in recent_tool_messages
            )
            system_message_content = (
                "You are an assistant that helps with course selection at Caltech. "
                "Use the following pieces of retrieved context about Caltech courses to "
                "answer the question. If you don't know the answer, say that you don't know. "
                "Do not answer questions that are irrelevant to Caltech courses, unless"
                "it is tangentially related to topics relating to courses that the user"
                "is inquiring about."
                f"{docs_content}"
            )

            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            response = self.llm.invoke(prompt)
            return {"messages": [response]}

        tools = ToolNode([retrieve])
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond", tools_condition, {END: END, "tools": "tools"}
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        self.graph = graph_builder.compile()

    def answer(self, input_message: str):
        state = {"messages": [{"role": "user", "content": input_message}]}
        final_state = self.graph.invoke(state)
        return final_state["messages"][-1].content
