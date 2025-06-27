from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from pydantic import BaseModel, Field

TOP_K = 1


class HybridSearchInput(BaseModel):
    query: str = Field(description="The query to search for.")


@tool(response_format="content_and_artifact", args_schema=HybridSearchInput)
def hybrid_search(query: str, vector_store: QdrantVectorStore):
    """
    Retrieve information related to a query about Caltech courses or
    related information, such as major/option requirements using past course
    reviews (student feedback) and the course catalog.
    """
    retrieved_docs = vector_store.similarity_search(query, k=TOP_K)
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
    print(list(doc_ids))
    return serialized, retrieved_docs


class CourseCatalogSearchInput(BaseModel):
    query: str = Field(
        description="The query to search for, which should include the course name and number."
    )
    depts: list[str] = Field(
        description="A list of department codes, e.g. ['CS', 'EE']"
    )
    code_number: int | None = Field(description="The course number, e.g. 101")


@tool(
    response_format="content_and_artifact",
    args_schema=CourseCatalogSearchInput,
)
def course_catalog_search(
    query: str,
    depts: list[str],
    vector_store: QdrantVectorStore,
    code_number: int | None = None,
):
    """
    Search for a specific Caltech course by department and course number using the course catalog.
    """
    if code_number is None:
        return "Error: A valid integer for code_number must be provided to use this tool."

    qdrant_filter = Filter(
        must=[
            FieldCondition(key="depts", match=MatchAny(any=depts)),
            FieldCondition(key="code_number", match=MatchValue(value=code_number)),
        ]
    )
    retrieved_docs = vector_store.similarity_search(
        query, k=TOP_K, filter=qdrant_filter
    )
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
    print(list(doc_ids))
    return serialized, retrieved_docs