from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

from rag import CourseRAG

if not load_dotenv():
    print("Unable to get environment variables via pydotenv.")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "caltech-courses"
index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

query = "Tell me about the machine learning courses."

query_vector = embedding_model.embed_query("CS 155")
results = index.query(vector=query_vector, top_k=10, include_metadata=True)

for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")

rag = CourseRAG(model_code="gpt-4o-mini", embedding_model="text-embedding-3-small")
print(rag.answer(input_message=query))
