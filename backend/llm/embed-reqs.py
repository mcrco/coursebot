from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
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

with open("json/reqs.json", "r") as f:
    data = json.load(f)

print("Creating documents for course catalog...")
documents = [
    {
        "id": f"{option.lower().replace(' ', '_')}",
        "content": (f"Graduation requirements for {option}: \n {option_text}"),
    }
    for option, option_text in tqdm(data.items())
]

print("Embedding documents...")
embedded_documents = [
    {
        "id": doc["id"],
        "embedding": embedding_model.embed_query(doc["content"]),
        "metadata": {
            "id": doc["id"],
            "source": "Caltech course catalog graduation requirements",
            "text": doc["content"],
        },
    }
    for doc in tqdm(documents)
]

print("Uploading documents...")
for doc in tqdm(embedded_documents):
    index.upsert(
        vectors=[
            {"id": doc["id"], "values": doc["embedding"], "metadata": doc["metadata"]}
        ]
    )

print(f"Uploaded {len(embedded_documents)} documents to Pinecone.")
