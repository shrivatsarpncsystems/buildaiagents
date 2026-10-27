import json

with open("students_results.json", "r", encoding="utf-8") as f:
    students = json.load(f)

names = set()
for s in students:
    for part in s["name"].split():
        names.add(part.lower())

print(f"Total students: {len(students)}")
print(f"Total unique name parts: {len(names)}")
print("Sample name parts:", sorted(list(names))[:30])

stop_words = {
    "and", "the", "for", "with", "from", "can", "you", "give", "show", 
    "what", "tell", "score", "scores", "marks", "mark", "grade", "grades",
    "sem", "semester", "semister", "semster", "student", "students", 
    "candidate", "candidates", "who", "which", "calculate", "calculatemarks",
    "percentage", "percent", "format", "formatt", "if", "ask", "particular",
    "every", "eery", "subject", "subjects", "individually", "indiidulally",
    "well", "total", "please", "how", "much", "did", "get", "got", "obtain",
    "obtained", "result", "results", "about", "also", "both", "all"
}

overlap = names.intersection(stop_words)
print("Overlap between student names and stop words:", overlap)
