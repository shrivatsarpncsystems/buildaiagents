import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

DATA_FILE = os.path.join(os.path.dirname(__file__), "students_results.json")

def _load_data() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file {DATA_FILE} not found. Please run generate_college_data.py first.")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_top_scorers(n: int = 5, semester: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve top N students by overall CGPA or by semester SGPA."""
    students = _load_data()
    n = max(1, min(n, 50))
    
    if semester is not None and 1 <= semester <= 8:
        sem_key = str(semester)
        sorted_students = sorted(
            students,
            key=lambda s: s["semesters"][sem_key]["sgpa"],
            reverse=True
        )[:n]
        results = []
        for idx, s in enumerate(sorted_students, 1):
            sem_data = s["semesters"][sem_key]
            results.append({
                "rank": idx,
                "usn": s["usn"],
                "name": s["name"],
                "sgpa": sem_data["sgpa"],
                "semester": sem_data["semester_name"],
                "overall_cgpa": s["cgpa"]
            })
        return {
            "type": "semester_toppers",
            "semester": semester,
            "count": len(results),
            "top_students": results
        }
    else:
        # Overall CGPA toppers
        sorted_students = sorted(students, key=lambda s: s["cgpa"], reverse=True)[:n]
        results = []
        for idx, s in enumerate(sorted_students, 1):
            results.append({
                "rank": s["rank"],
                "usn": s["usn"],
                "name": s["name"],
                "cgpa": s["cgpa"],
                "standing": s["standing"],
                "backlogs": s["backlogs"]
            })
        return {
            "type": "overall_cgpa_toppers",
            "count": len(results),
            "top_students": results
        }

def find_student(identifier: str) -> Optional[Dict[str, Any]]:
    """Find a student by USN (e.g. 1RV21CS042) or by full/partial name."""
    students = _load_data()
    q = identifier.strip().lower()
    
    # 1. Exact USN match
    for s in students:
        if s["usn"].lower() == q:
            return s
            
    # 2. Exact or substring Name match
    matches = [s for s in students if q in s["name"].lower()]
    if matches:
        return matches[0]
        
    return None

def find_students_by_name(name: str) -> List[Dict[str, Any]]:
    """Find all students matching a full or partial name (handles name collisions)."""
    students = _load_data()
    q = name.strip().lower()
    if not q:
        return []
    return [s for s in students if q in s["name"].lower()]

def get_student_scorecard(identifier: str) -> Dict[str, Any]:
    """Get full 8-semester scorecard and analytics for a student."""
    student = find_student(identifier)
    if not student:
        return {"error": f"Student '{identifier}' not found in college examination records."}
        
    sgpas = [student["semesters"][str(sm)]["sgpa"] for sm in range(1, 9)]
    best_sem = sgpas.index(max(sgpas)) + 1
    worst_sem = sgpas.index(min(sgpas)) + 1
    
    # Calculate total subjects and total marks
    all_subjects = []
    for sm in range(1, 9):
        for sub in student["semesters"][str(sm)]["subjects"]:
            all_subjects.append(sub)
            
    avg_marks = round(sum(s["marks"] for s in all_subjects) / len(all_subjects), 2)
    highest_subj = max(all_subjects, key=lambda x: x["marks"])
    lowest_subj = min(all_subjects, key=lambda x: x["marks"])
    
    return {
        "usn": student["usn"],
        "name": student["name"],
        "rank": student["rank"],
        "branch": student["branch"],
        "batch": student["batch"],
        "cgpa": student["cgpa"],
        "standing": student["standing"],
        "backlogs": student["backlogs"],
        "total_credits": student["total_credits"],
        "sgpa_by_semester": {f"Sem {sm}": student["semesters"][str(sm)]["sgpa"] for sm in range(1, 9)},
        "best_semester": f"Sem {best_sem} (SGPA: {max(sgpas):.2f})",
        "lowest_semester": f"Sem {worst_sem} (SGPA: {min(sgpas):.2f})",
        "average_marks": avg_marks,
        "highest_subject": f"{highest_subj['name']} ({highest_subj['marks']} marks)",
        "lowest_subject": f"{lowest_subj['name']} ({lowest_subj['marks']} marks)",
        "semesters_detail": student["semesters"]
    }

def calculate_batch_analytics(semester: Optional[int] = None) -> Dict[str, Any]:
    """Calculate average, highest, lowest, pass rate for the batch or a specific semester."""
    students = _load_data()
    total_students = len(students)
    
    if semester is not None and 1 <= semester <= 8:
        sem_key = str(semester)
        sgpas = [s["semesters"][sem_key]["sgpa"] for s in students]
        avg_sgpa = round(sum(sgpas) / total_students, 2)
        max_sgpa = max(sgpas)
        min_sgpa = min(sgpas)
        passed = sum(1 for s in students if not any(sub["grade"] == "F" for sub in s["semesters"][sem_key]["subjects"]))
        
        return {
            "type": "semester_batch_stats",
            "semester": semester,
            "total_candidates": total_students,
            "average_sgpa": avg_sgpa,
            "highest_sgpa": max_sgpa,
            "lowest_sgpa": min_sgpa,
            "pass_percentage": round((passed / total_students) * 100, 2),
            "students_passed": passed,
            "students_with_backlogs": total_students - passed
        }
    else:
        cgpas = [s["cgpa"] for s in students]
        avg_cgpa = round(sum(cgpas) / total_students, 2)
        max_cgpa = max(cgpas)
        min_cgpa = min(cgpas)
        distinction = sum(1 for c in cgpas if c >= 8.5)
        first_class = sum(1 for c in cgpas if 6.75 <= c < 8.5)
        second_class = sum(1 for c in cgpas if 5.0 <= c < 6.75)
        backlog_count = sum(1 for s in students if s["backlogs"] > 0)
        
        # Semester-by-semester average trend
        sem_averages = {}
        for sm in range(1, 9):
            s_avg = round(sum(s["semesters"][str(sm)]["sgpa"] for s in students) / total_students, 2)
            sem_averages[f"Sem {sm}"] = s_avg
            
        return {
            "type": "overall_batch_stats",
            "total_candidates": total_students,
            "average_cgpa": avg_cgpa,
            "highest_cgpa": max_cgpa,
            "lowest_cgpa": min_cgpa,
            "distinction_count": distinction,
            "distinction_pct": round(distinction / total_students * 100, 2),
            "first_class_count": first_class,
            "first_class_pct": round(first_class / total_students * 100, 2),
            "second_class_count": second_class,
            "candidates_with_backlogs": backlog_count,
            "semester_sgpa_averages": sem_averages
        }

DOMAIN_SUBJECT_MAPPINGS = {
    # Mathematics
    "maths": ["Engineering Mathematics I", "Engineering Mathematics II", "Discrete Mathematical Structures"],
    "math": ["Engineering Mathematics I", "Engineering Mathematics II", "Discrete Mathematical Structures"],
    "mathematics": ["Engineering Mathematics I", "Engineering Mathematics II", "Discrete Mathematical Structures"],
    "discrete math": ["Discrete Mathematical Structures"],
    "discrete mathematical": ["Discrete Mathematical Structures"],
    "discrete": ["Discrete Mathematical Structures"],
    "math 1": ["Engineering Mathematics I"],
    "maths 1": ["Engineering Mathematics I"],
    "math 2": ["Engineering Mathematics II"],
    "maths 2": ["Engineering Mathematics II"],

    # Core CS Theory & Systems
    "data structures": ["Data Structures & Applications", "Data Structures Laboratory"],
    "dsa": ["Data Structures & Applications", "Data Structures Laboratory"],
    "algorithms": ["Design & Analysis of Algorithms", "Algorithms Laboratory"],
    "daa": ["Design & Analysis of Algorithms", "Algorithms Laboratory"],
    "operating systems": ["Operating Systems"],
    "os": ["Operating Systems"],
    "database": ["Database Management Systems", "DBMS Laboratory"],
    "dbms": ["Database Management Systems", "DBMS Laboratory"],
    "sql": ["Database Management Systems", "DBMS Laboratory"],
    "computer networks": ["Computer Networks", "Computer Networks Lab"],
    "networking": ["Computer Networks", "Computer Networks Lab"],
    "networks": ["Computer Networks", "Computer Networks Lab"],
    "cn": ["Computer Networks", "Computer Networks Lab"],
    "compiler": ["Compiler Design"],
    "compiler design": ["Compiler Design"],
    "theory of computation": ["Theory of Computation & Automata"],
    "automata": ["Theory of Computation & Automata"],
    "toc": ["Theory of Computation & Automata"],
    "computer organization": ["Computer Organization & Architecture"],
    "coa": ["Computer Organization & Architecture"],
    "software engineering": ["Software Engineering & Agile"],
    "agile": ["Software Engineering & Agile"],
    "web development": ["Web Technologies & Frameworks", "Web Development Lab"],
    "web technologies": ["Web Technologies & Frameworks", "Web Development Lab"],
    "web tech": ["Web Technologies & Frameworks", "Web Development Lab"],
    "web": ["Web Technologies & Frameworks", "Web Development Lab"],
    "microprocessors": ["Microprocessors & Embedded Systems"],
    "embedded systems": ["Microprocessors & Embedded Systems"],

    # Advanced AI, Data & Security
    "machine learning": ["Machine Learning & Pattern Recognition", "Machine Learning Laboratory"],
    "ml": ["Machine Learning & Pattern Recognition", "Machine Learning Laboratory"],
    "artificial intelligence": ["Artificial Intelligence & Expert Systems", "AI & Intelligence Lab"],
    "ai": ["Artificial Intelligence & Expert Systems", "AI & Intelligence Lab"],
    "deep learning": ["Deep Learning & Generative AI"],
    "generative ai": ["Deep Learning & Generative AI"],
    "genai": ["Deep Learning & Generative AI"],
    "gen ai": ["Deep Learning & Generative AI"],
    "big data": ["Big Data Analytics & Spark"],
    "spark": ["Big Data Analytics & Spark"],
    "nlp": ["Elective I: Natural Language Processing"],
    "natural language processing": ["Elective I: Natural Language Processing"],
    "iot": ["Internet of Things & Robotics"],
    "robotics": ["Internet of Things & Robotics"],
    "cloud": ["Cloud Computing & Virtualization"],
    "cloud computing": ["Cloud Computing & Virtualization"],
    "cryptography": ["Cryptography & Network Security", "Security & Pentesting Lab"],
    "security": ["Cryptography & Network Security", "Security & Pentesting Lab"],
    "pentesting": ["Security & Pentesting Lab"],
    "blockchain": ["Distributed Systems & Blockchain"],
    "distributed systems": ["Distributed Systems & Blockchain"],
    "devops": ["Elective II: DevOps & Microservices"],
    "microservices": ["Elective II: DevOps & Microservices"],

    # Programming Languages
    "java": ["Object Oriented Programming with Java", "Java Laboratory"],
    "oop": ["Object Oriented Programming with Java", "Java Laboratory"],
    "c programming": ["C Programming & Problem Solving", "C Programming Lab"],
    "c language": ["C Programming & Problem Solving", "C Programming Lab"],

    # Sciences / Projects
    "physics": ["Engineering Physics", "Engineering Physics Lab"],
    "chemistry": ["Engineering Chemistry", "Engineering Chemistry Lab"],
    "english": ["Technical English"],
    "electrical": ["Basic Electrical Engineering"],
    "electronics": ["Basic Electronics Engineering"],
    "graphics": ["Engineering Graphics"],
    "project": ["Major Project Phase I", "Major Project Phase II"],
    "major project": ["Major Project Phase I", "Major Project Phase II"],
    "seminar": ["Technical Seminar & Paper Publication"],
}

def get_subject_or_domain_analytics(query_text: str) -> Optional[str]:
    """
    Intelligently identifies if a query asks about a specific academic course or subject domain
    (e.g. 'Who is best at maths overall', 'Topper in Machine Learning', 'Average score in DBMS').
    Computes exact mathematical aggregation, course-by-course highlights, and top scorers.
    """
    students = _load_data()
    q_clean = query_text.lower().strip()
    
    matched_subjects = []
    domain_name = ""
    
    # 1. Match against domain mappings (sorted by length descending)
    for k in sorted(DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
        if re.search(pattern, q_clean):
            matched_subjects = DOMAIN_SUBJECT_MAPPINGS[k]
            domain_name = k.title()
            break
            
    # 2. Match directly against any course name or code in the curriculum
    if not matched_subjects:
        for sm in range(1, 9):
            for sub in students[0]["semesters"][str(sm)]["subjects"]:
                if sub["name"].lower() in q_clean or sub["code"].lower() in q_clean:
                    matched_subjects = [sub["name"]]
                    domain_name = sub["name"]
                    break
            if matched_subjects:
                break
                
    if not matched_subjects:
        return None
        
    out = []
    out.append(f"[OFFICIAL SUBJECT & DOMAIN ACADEMIC ANALYTICS: {domain_name.upper()}]")
    out.append(f"Curriculum Scope: {len(matched_subjects)} Course(s) across all 200 Candidates in CSE Cohort")
    out.append(f"Courses Evaluated: {', '.join(matched_subjects)}\n")
    
    # 3. Overall Domain Aggregation (when domain has multiple courses, e.g. Mathematics, AI, Labs)
    if len(matched_subjects) > 1:
        student_aggs = []
        for s in students:
            scores = []
            for sm in s["semesters"].values():
                for sub in sm["subjects"]:
                    if sub["name"] in matched_subjects:
                        scores.append({"name": sub["name"], "marks": sub["marks"], "grade": sub["grade"]})
            if scores:
                avg_m = round(sum(x["marks"] for x in scores) / len(scores), 2)
                student_aggs.append({"usn": s["usn"], "name": s["name"], "avg": avg_m, "scores": scores, "cgpa": s["cgpa"]})
                
        student_aggs.sort(key=lambda x: x["avg"], reverse=True)
        
        out.append(f"TOP PERFORMERS OVERALL IN {domain_name.upper()} (Combined Average across all {len(matched_subjects)} courses):")
        for idx, st in enumerate(student_aggs[:5], 1):
            sc_details = ", ".join([f"{sc['name']}: {sc['marks']}/100 ({sc['grade']})" for sc in st["scores"]])
            out.append(f"Rank #{idx}: {st['name']} (USN: {st['usn']}) - Combined Average: {st['avg']}% | CGPA: {st['cgpa']:.2f} [{sc_details}]")
        out.append("")
        
    # 4. Course-by-Course Detailed Performance and Toppers
    out.append("COURSE-BY-COURSE DETAILED PERFORMANCE & TOPPERS:")
    for sub_name in matched_subjects:
        all_scores = []
        sub_code = ""
        sub_sem = ""
        for s in students:
            for sm_idx, sm in s["semesters"].items():
                for sub in sm["subjects"]:
                    if sub["name"] == sub_name:
                        sub_code = sub["code"]
                        sub_sem = sm_idx
                        all_scores.append({"usn": s["usn"], "name": s["name"], "marks": sub["marks"], "grade": sub["grade"]})
                        
        if not all_scores:
            continue
            
        all_scores.sort(key=lambda x: x["marks"], reverse=True)
        marks_vals = [x["marks"] for x in all_scores]
        avg_score = round(sum(marks_vals) / len(marks_vals), 2)
        max_score = marks_vals[0]
        min_score = marks_vals[-1]
        toppers = [x for x in all_scores if x["marks"] == max_score]
        fail_cnt = sum(1 for m in marks_vals if m < 40)
        pass_pct = round(((len(marks_vals) - fail_cnt) / len(marks_vals)) * 100, 2)
        
        top_str = ", ".join([f"{t['name']} (USN: {t['usn']})" for t in toppers[:3]])
        out.append(f"- Course: {sub_name} (Code: {sub_code}, Semester {sub_sem})")
        out.append(f"  * Highest Score Achieved: {max_score}/100 by: {top_str}")
        out.append(f"  * Class Average: {avg_score}/100 | Lowest Score: {min_score}/100 | Pass Rate: {pass_pct}% ({fail_cnt} backlogs)")
        top_3_str = ", ".join([f"#{i} {st['name']} ({st['usn']}): {st['marks']}/100 ({st['grade']})" for i, st in enumerate(all_scores[:3], 1)])
        out.append(f"  * Top 3 Scorers: {top_3_str}")
        
    return "\n".join(out)

def get_subject_analytics(subject_query: str) -> Dict[str, Any]:
    """Backward-compatible analytics helper."""
    report = get_subject_or_domain_analytics(subject_query)
    return {"report": report} if report else {"error": f"No subject matching '{subject_query}' found."}

def compare_students(id1: str, id2: str) -> Dict[str, Any]:
    """Compare two students side-by-side."""
    st1 = find_student(id1)
    st2 = find_student(id2)
    
    if not st1:
        return {"error": f"Student '{id1}' not found."}
    if not st2:
        return {"error": f"Student '{id2}' not found."}
        
    sem_comparison = []
    for sm in range(1, 9):
        sg1 = st1["semesters"][str(sm)]["sgpa"]
        sg2 = st2["semesters"][str(sm)]["sgpa"]
        diff = round(sg1 - sg2, 2)
        sem_comparison.append({
            "semester": f"Sem {sm}",
            f"{st1['name']} ({st1['usn']})": sg1,
            f"{st2['name']} ({st2['usn']})": sg2,
            "higher": f"{st1['name']} ({st1['usn']})" if sg1 > sg2 else (f"{st2['name']} ({st2['usn']})" if sg2 > sg1 else "Tie"),
            "difference": abs(diff)
        })
        
    winner = f"{st1['name']} ({st1['usn']})" if st1["cgpa"] > st2["cgpa"] else (f"{st2['name']} ({st2['usn']})" if st2["cgpa"] > st1["cgpa"] else "Tie")
    return {
        "student1": {"usn": st1["usn"], "name": st1["name"], "rank": st1["rank"], "cgpa": st1["cgpa"], "standing": st1["standing"], "backlogs": st1["backlogs"], "total_credits": st1["total_credits"]},
        "student2": {"usn": st2["usn"], "name": st2["name"], "rank": st2["rank"], "cgpa": st2["cgpa"], "standing": st2["standing"], "backlogs": st2["backlogs"], "total_credits": st2["total_credits"]},
        "cgpa_difference": round(abs(st1["cgpa"] - st2["cgpa"]), 2),
        "cgpa_winner": winner,
        "semesters_comparison": sem_comparison
    }

def get_performance_list(category: str, limit: int = 20) -> Dict[str, Any]:
    """Get list of students by distinction, first class, or active backlogs."""
    students = _load_data()
    cat = category.strip().lower()
    
    if "distinction" in cat:
        matched = [s for s in students if s["cgpa"] >= 8.5]
        label = "Distinction Achievers (CGPA >= 8.5)"
    elif "first" in cat:
        matched = [s for s in students if 6.75 <= s["cgpa"] < 8.5]
        label = "First Class Achievers (6.75 <= CGPA < 8.5)"
    elif "backlog" in cat or "fail" in cat:
        matched = [s for s in students if s["backlogs"] > 0]
        label = "Candidates with Active Backlogs"
    else:
        matched = students[:limit]
        label = f"Students list (first {limit})"
        
    return {
        "category": label,
        "total_in_category": len(matched),
        "students": [{
            "rank": s["rank"],
            "usn": s["usn"],
            "name": s["name"],
            "cgpa": s["cgpa"],
            "standing": s["standing"],
            "backlogs": s["backlogs"]
        } for s in matched[:limit]]
    }

def get_failed_students(semester: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve list of students who failed / have backlogs in a specific semester or overall."""
    students = _load_data()
    failed_students = []
    
    if semester is not None and 1 <= semester <= 8:
        sem_key = str(semester)
        for s in students:
            sem_data = s["semesters"][sem_key]
            failed_subs = [sub for sub in sem_data["subjects"] if sub["grade"] == "F"]
            if failed_subs:
                failed_students.append({
                    "usn": s["usn"],
                    "name": s["name"],
                    "semester": semester,
                    "sgpa": sem_data["sgpa"],
                    "overall_cgpa": s["cgpa"],
                    "failed_subjects": [f"{sub['name']} ({sub['marks']}/100, Grade: F)" for sub in failed_subs]
                })
        return {
            "type": "semester_failed_students",
            "semester": semester,
            "count": len(failed_students),
            "students": failed_students
        }
    else:
        for s in students:
            if s["backlogs"] > 0:
                failed_students.append({
                    "usn": s["usn"],
                    "name": s["name"],
                    "cgpa": s["cgpa"],
                    "backlogs_count": s["backlogs"],
                    "standing": s["standing"]
                })
        return {
            "type": "overall_failed_students",
            "count": len(failed_students),
            "students": failed_students
        }
