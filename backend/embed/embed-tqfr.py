import itertools
from fastembed import SparseTextEmbedding
from langchain_google_vertexai import VertexAIEmbeddings
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


with open("json/tqfr.json", "r") as f:
    data = json.load(f)


def score2text(score, avg):
    score, avg = float(score), float(avg)
    frac_diff = float("inf")
    if avg != 0:
        frac_diff = abs((score - avg) / avg)
    degree = None
    if frac_diff < 0.1:
        degree = "a bit"
    elif frac_diff >= 0.3:
        degree = "a lot"

    if score < avg:
        comp = "lower than"
    elif score > avg:
        comp = "higher than"
    else:
        comp = "equal to"

    if degree:
        return degree + " " + comp
    return comp


def process_table(question, data, instructor=None):
    about_text = f" about {instructor}, " if instructor else ""
    return (
        f'When asked "{question.lower()}"'
        + about_text
        + ", ".join(
            [
                f"{percent} of student responses said {option}"
                for option, percent in data.items()
            ]
        )
        + "."
    )


def process_score(question, data, instructor=None):
    score, stdev, dept_avg, caltech_avg = (
        data["score"],
        data["stdev"],
        data["dept"],
        data["caltech"],
    )

    about_text = f" about {instructor}" if instructor else ""
    return (
        f'When asked "{question.lower()}"' + about_text + ", students on average rated "
        f'{data['score']} out of 5 with a standard deviation of {stdev},'
        f"which was {score2text(score, dept_avg)} than the department average of {dept_avg} "
        f"and {score2text(score, caltech_avg)} the Caltech average of {caltech_avg}. "
    )


with open("json/tqfr.json", "r") as f:
    data = json.load(f)

print("Creating chunks for TQFRs...")
ids = []
metas = []
summaries = []
contents = []
for key in tqdm(data):
    for term in data[key]:
        report = data[key][term]
        course_id, name = report["course_id"], report["name"]

        course_qas = []
        for question, response_data in report["course"].items():
            if "score" in response_data:
                course_qas.append(process_score(question, response_data))
            else:
                course_qas.append(process_table(question, response_data))

        report_id = f"{key}-{term.lower().replace('-', '_').replace(' ', '-')}-tqfr"
        source = f"TQFR {term} for {report['course_id']}: {name}"
        course_chunk_id = f"{report_id}-course"
        course_chunk_content = (
            f"Course-related feedback during {term} for {course_id}: {name}:\n"
            + f"Response rate was {report['response_rate']}\n"
            + "\n".join(course_qas)
        )
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, course_chunk_id)))
        metas.append(
            {
                "url": report["url"],
                "source": source,
                "text": report["raw_text"],
                "dense_model": dense_model,
                "sparse_model": sparse_model,
                "doc_id": report_id,
            }
        )
        summaries.append(
            f"Student feedback for {course_id}: {name} during the {term} term"
        )
        contents.append(course_chunk_content)

        if "instructor" in report:
            for instructor, inst_data in report["instructor"].items():
                inst_qas = []
                for question, response_data in inst_data.items():
                    if question == "type":
                        continue
                    if "score" in response_data:
                        inst_qas.append(
                            process_score(question, response_data, instructor)
                        )
                    else:
                        inst_qas.append(
                            process_table(question, response_data, instructor)
                        )

                chunk_id = f"{report_id}-{"-".join([name.lower() for name in instructor.split()])}"
                chunk_content = (
                    f"{inst_data['type']} feedback for {instructor} during {term} for {course_id}: {name}:\n"
                    + f"Response rate was {report['response_rate']}\n"
                    + "\n".join(inst_qas)
                )
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)))
                metas.append(
                    {
                        "url": report["url"],
                        "source": source,
                        "text": report["raw_text"],
                        "dense_model": dense_model,
                        "sparse_model": sparse_model,
                        "doc_id": report_id,
                    }
                )
                summaries.append(
                    f"Student feedback for {instructor} teaching {course_id}: {name} during the {term} term"
                )
                contents.append(chunk_content)

        comment_chunk_id = f"{report_id}-comments"
        comment_chunk_content = (
            f"Comments/advice from students who took {course_id}: {name} during {term}:\n"
            + "\n".join(report["comments"])
        )
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, comment_chunk_id)))
        metas.append(
            {
                "url": report["url"],
                "source": source,
                "text": report["raw_text"],
                "dense_model": dense_model,
                "sparse_model": sparse_model,
                "doc_id": report_id,
            }
        )
        summaries.append(
            f"Comments/advice from students who took {course_id}: {name} during {term}"
        )
        contents.append(comment_chunk_content)


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
for batch in itertools.batched(points, 512):
    vector_db.upsert(collection_name=collection_name, points=list(batch))

print(f"Uploaded {len(points)} points to Qdrant.")
