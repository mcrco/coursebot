import json
import os

merged = {}

for fname in os.listdir("../json/"):
    if fname.endswith("json"):
        fpath = os.path.join("../json/", fname)
        with open(fpath, "r") as f:
            courses = json.load(f)
            for id, course in courses.items():
                if id not in merged:
                    merged[id] = course
                else:
                    for prereq_id in course["prereqs"]:
                        if prereq_id not in merged[id]["prereqs"]:
                            merged[id]["prereqs"].append(prereq_id)

with open("json/courses.json", "w") as f:
    json.dump(merged, f, indent=4)
