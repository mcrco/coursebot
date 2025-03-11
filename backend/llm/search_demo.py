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


query = "Tell me about cs 155"

results = vector_store.similarity_search_with_score(query, k=3)

for match in results:
    print(match)
