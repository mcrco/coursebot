from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json
import os

if not load_dotenv():
    print("Unable to get environment variables via pydotenv.")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "caltech-courses"
embedding_dimensions = 1536

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimensions,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

with open("json/courses.json", "r") as f:
    data = json.load(f)

documents = [
    {
        "id": course_id,
        "content": "\n".join(f"{k}: {v}" for k, v in course.items()),
    }
    for course_id, course in data.items()
]

embedded_documents = [
    {
        "id": doc["id"],
        "embedding": embedding_model.embed_query(doc["content"]),
        "metadata": {
            "id": doc["id"],
            "source": "caltech-courses",
            "text": doc["content"],
        },  # Include metadata
    }
    for doc in documents
]

for doc in embedded_documents:
    index.upsert(
        vectors=[
            {"id": doc["id"], "values": doc["embedding"], "metadata": doc["metadata"]}
        ]
    )

print(f"Uploaded {len(embedded_documents)} documents to Pinecone.")

query_vector = embedding_model.embed_query("Tell me about CS 155.")
results = index.query(vector=query_vector, top_k=5, include_metadata=True)

for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")
