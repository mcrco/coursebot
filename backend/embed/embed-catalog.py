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
dense_embed = VertexAIEmbeddings(model=dense_model, project="coursebot-453309")
sparse_embed = SparseTextEmbedding(sparse_model)

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
summaries = []
contents = []
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
        ids.append(id)
        metas.append(
            {
                "url": entry["url"],
                "source": entry["source"],
                "text": headers + text,
                "dense_model": dense_model,
                "sparse_model": sparse_model,
                "doc_id": id,
            }
        )
        summaries.append(headers)
        contents.append(headers + text)

print("Embedding document chunks...")
dense_summaries = dense_embed.embed(summaries)
sparse_summmaries = sparse_embed.embed(summaries)
dense_contents = dense_embed.embed(contents)
sparse_contents = sparse_embed.embed(contents)


def id2uuid(id):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id))


print("Creating points...")
points = []
for id, ds, ss, dc, sc, meta in tqdm(
    zip(ids, dense_summaries, sparse_summmaries, dense_contents, sparse_contents, metas)
):
    points.append(
        PointStruct(
            id=id2uuid(f"{id}-summary"),
            vector={
                "dense_vector": ds,
                "sparse_vector": vars(ss),
            },
            payload={"metadata": meta},
        )
    )
    points.append(
        PointStruct(
            id=id2uuid(f"{id}-content"),
            vector={
                "dense_vector": dc,
                "sparse_vector": vars(sc),
            },
            payload={"metadata": meta},
        )
    )

print("Uploading to Qdrant...")
for batch in itertools.batched(points, 1024):
    vector_db.upsert(collection_name=collection_name, points=list(batch))

print(f"Uploaded {len(points)} points to Qdrant.")
