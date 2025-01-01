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
embed_dim = 1536


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embed_dim,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


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
chunks = []
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

        course_chunk_id = f"{key}-tqfr-course"
        course_chunk_content = (
            f"Course-related feedback during {term} for {course_id}: {name}:\n"
            + f"Response rate was {report['response_rate']}\n"
            + "\n".join(course_qas)
        )

        chunks.append(
            {
                "id": course_chunk_id,
                "content": course_chunk_content,
                "metadata": {
                    "id": course_chunk_id,
                    "source": f"TQFR {term} for {course_id}: {name}",
                    "text": course_chunk_content,
                },
            }
        )

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

                chunk_id = f"{key}-tqfr-{"-".join([name.lower() for name in instructor.split()])}"
                chunk_content = (
                    f"{inst_data['type']} feedback for {instructor} during {term} for {course_id}: {name}:\n"
                    + f"Response rate was {report['response_rate']}\n"
                    + "\n".join(inst_qas)
                )
                chunks.append(
                    {
                        "id": chunk_id,
                        "content": chunk_content,
                        "metadata": {
                            "id": chunk_id,
                            "source": f"TQFR {term} for {course_id}: {name}",
                            "text": chunk_content,
                        },
                    }
                )

        comment_chunk_id = f"{key}-tqfr-comments"
        comment_chunk_content = (
            f"Comments/advice from students who took {course_id}: {name} during {term}:\n"
            + "\n".join(report["comments"])
        )
        chunks.append(
            {
                "id": comment_chunk_id,
                "content": comment_chunk_content,
                "metadata": {
                    "id": comment_chunk_id,
                    "source": f"TQFR {term} for {course_id}: {name}",
                    "text": comment_chunk_content,
                },
            }
        )


print("Embedding chunks for TQFRs...")
embedded_chunks = [
    {
        "id": chunk["id"],
        "embedding": embedding_model.embed_query(chunk["content"]),
        "metadata": {
            "id": chunk["id"],
            "source": "TQFR",
            "text": chunk["content"],
        },
    }
    for chunk in tqdm(chunks)
]

print("Uploading chunks for TQFRs...")
for chunk in embedded_chunks:
    index.upsert(
        vectors=[
            {
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": chunk["metadata"],
            }
        ]
    )

print(f"Uploaded {len(chunks)} vectors to Pinecone.")
