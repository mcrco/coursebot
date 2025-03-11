from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone
from dotenv import load_dotenv
import os


class CourseRAG:
    def __init__(
        self,
        model_code="deepseek-chat",
        embedding_model="models/text-embedding-004",
        sparse_embedding_model="Qdrant/bm25",
    ):
        if not load_dotenv():
            print("Unable to get environment variables via pydotenv.")

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index("caltech-courses")

        if "deepseek" in model_code:
            self.llm = ChatOpenAI(
                model=model_code,
                openai_api_key=os.environ["DEEPSEEK_API_KEY"],
                openai_api_base="https://api.deepseek.com",
                temperature=0.0,
            )
        elif "gemini" in model_code:
            self.llm = ChatGoogleGenerativeAI(
                model=model_code,
                temperature=0.0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        else:
            raise Exception("invalid llm model code")

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
            retrieval_mode=RetrievalMode.HYBRID,
        )

        self.build_graph()

    def build_graph(self):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """
            Retrieve information related to a query about Caltech courses or
            related information, such as major/option requirements using past course
            reviews (student feedback) and the course catalog.
            """
            retrieved_docs = self.vector_store.similarity_search(query, k=8)
            serialized = []
            doc_ids = set()
            for doc in retrieved_docs:
                doc_id = doc.metadata["_id"]
                if "doc_id" in doc.metadata:
                    doc_id = doc.metadata["doc_id"]
                if doc_id not in doc_ids:
                    doc_ids.add(doc_id)
                    serialized.append(
                        f"Source: {doc.metadata['source']}\nLink: {doc.metadata['url']}\nContent: {doc.metadata['text']}\n\n\n"
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
            system_message_content = """You are an AI assistant designed to help users with course selection at Caltech. You have access to a database of Caltech course information, professor details, and Teaching Quality Feedback Reports (TQFRs) via a retrieval tool.

                ### Guidelines:

                1. **Scope Restriction:**
                - Only answer questions related to Caltech courses, professors, scheduling, and related academic topics.
                - If a question is outside this scope, politely inform the user that you can only assist with Caltech course selection.

                2. **Preventing Hallucination:**
                - Your responses must be strictly based on retrieved context.
                - If you lack sufficient retrieved information to answer, say: _"I couldn’t find relevant information in the Caltech course catalog and TQFRs from the past two years."_
                - Do not attempt to infer or speculate beyond the provided data.

                3. **Mandatory Citation:**
                - Any statement derived from retrieved data must be followed by a markdown link in parentheses formatted as: **[Source Name](Source URL)**
                - Ensure citations are specific and clearly support the information provided.
                - If source names are really long and there are a lot of them, shorten them to an abbreviation.

                4. **Prohibition on Code Generation:**
                - Do not generate or suggest custom code, scripts, or programming solutions.
                - If a user asks for code, respond: _"I am designed to assist with Caltech course selection and cannot provide code."_

                5. **TQFRs:**
                - If you are answering based on a TQFR, include the tables in the retrieved markdown for user friendliness.
                - If there are a lot of TQFRs, generate summaries for each question based on the student responses, you don't have to include the tables.
                - It's always nice to include quotes from the student comments section if there are any.
            """

            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = (
                [SystemMessage(system_message_content)]
                + conversation_messages
                + [f"\n Retrieved context: {docs_content}"]
                + [
                    """
                    Keep in mind your original instructions!
                    You don't have to use all of the retrieved documents; just pick out the ones relevant to the last user question and answer based off of those.
                    Note that you are limited to 8000 output tokens, so try to keep the response within that margin.
                    """
                ]
            )
            print(prompt)
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

    def complete(self, messages):
        state = {"messages": messages}
        final_state = self.graph.invoke(state)
        return final_state["messages"][-1]

    def stream_complete(self, messages):
        state = {"messages": messages}
        for chunk in self.graph.stream(state, stream_mode=["messages"]):
            _, (message, metadata) = chunk
            if metadata["langgraph_node"] in ["query_or_respond", "generate"]:
                yield message
