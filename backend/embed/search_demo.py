from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
import os

if not load_dotenv():
    print("Unable to get environment variables via pydotenv.")

embeddings = VertexAIEmbeddings(model="text-embedding-004", project="coursebot-453309")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
vector_store = QdrantVectorStore.from_existing_collection(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name="coursebot_hybrid",
    content_payload_key="text",
    metadata_payload_key="metadata",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    vector_name="dense_vector",
    sparse_vector_name="sparse_vector",
    retrieval_mode=RetrievalMode.DENSE,
)


while True:
    query = input("Enter query: ")
    n_results = int(input("Number of results to list: "))
    results = vector_store.similarity_search_with_score(query, k=3)

    for doc, sim in results:
        print(f"Matched {doc.metadata['doc_id']} with similarity {sim}")
