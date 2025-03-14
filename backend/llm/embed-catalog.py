import itertools
from fastembed import SparseTextEmbedding
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)
from dotenv import load_dotenv
from tqdm import tqdm
import json
import os
import string
import uuid

if not load_dotenv():
    print("Unable to get environment variables via pydotenv.")

vector_db = QdrantClient(
    url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], port=None
)
dense_model = "text-embedding-004"
sparse_model = "Qdrant/bm25"
dense_embed = VertexAIEmbeddings(model="text-embedding-004", project="coursebot-453309")
sparse_embed = SparseTextEmbedding("Qdrant/bm25")

collection_name = "coursebot_hybrid"
if not vector_db.collection_exists(collection_name):
    vector_db.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense_vector": VectorParams(
                size=768, distance=Distance.COSINE
            )  # 768 is dim for google embedding model
        },
        sparse_vectors_config={"sparse_vector": SparseVectorParams()},
    )


with open("json/catalog.json", "r") as f:
    data = json.load(f)

print("Chunking documents...")
headers_to_split_on = [
    ("##", "h2"),
    ("###", "h3"),
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
ids = []
metas = []
texts = []
for entry in tqdm(data.values()):
    content = entry["content"]
    splits = text_splitter.split_text(content)
    for chunk in splits:
        text = chunk.page_content
        id = entry["title"]
        headers = "# " + entry["title"] + "\n\n"
        if "h2" in chunk.metadata:
            headers += "## " + chunk.metadata["h2"] + "\n\n"
            id += " " + chunk.metadata["h2"]
        if "h3" in chunk.metadata:
            headers += "### " + chunk.metadata["h3"] + "\n\n"
            id += " " + chunk.metadata["h3"]
        id = (
            id.lower()
            .translate(str.maketrans("", "", string.punctuation))
            .replace(" ", "_")
        )
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, id)))
        metas.append(
            {
                "url": entry["url"],
                "source": entry["source"],
                "text": headers + text,
                "dense_model": dense_model,
                "sparse_model": sparse_model,
            }
        )
        texts.append(headers + text)

print("Embedding document chunks...")
dense_vectors = dense_embed.embed(texts)
sparse_vectors = sparse_embed.embed(texts)

print("Creating points...")
points = []
for id, dense, sparse, meta in tqdm(zip(ids, dense_vectors, sparse_vectors, metas)):
    points.append(
        PointStruct(
            id=id,
            vector={
                "dense_vector": dense,
                "sparse_vector": {
                    "indices": list(sparse.indices),
                    "values": list(sparse.values),
                },
            },
            payload={"metadata": meta},
        )
    )

print("Uploading to Qdrant...")
for batch in itertools.batched(points, 1024):
    vector_db.upsert(collection_name=collection_name, points=list(batch))

print(f"Uploaded {len(points)} points to Qdrant.")
