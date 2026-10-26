import json
import os
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

def get_subject_analytics(subject_query: str) -> Dict[str, Any]:
    """Analyze subject scores across all students."""
    students = _load_data()
    q = subject_query.strip().lower()
    
    # Collect all subject entries matching query
    matching_subject_name = None
    all_scores = []
    
    for s in students:
        for sm in range(1, 9):
            for sub in s["semesters"][str(sm)]["subjects"]:
                if q in sub["name"].lower() or q in sub["code"].lower():
                    if matching_subject_name is None:
                        matching_subject_name = sub["name"]
                    all_scores.append({
                        "student_usn": s["usn"],
                        "student_name": s["name"],
                        "marks": sub["marks"],
                        "grade": sub["grade"],
                        "semester": sm
                    })
                    
    if not all_scores:
        return {"error": f"No subject matching '{subject_query}' found in curriculum."}
        
    marks_list = [entry["marks"] for entry in all_scores]
    avg_marks = round(sum(marks_list) / len(marks_list), 2)
    max_marks = max(marks_list)
    min_marks = min(marks_list)
    toppers = [entry for entry in all_scores if entry["marks"] == max_marks][:3]
    fail_count = sum(1 for m in marks_list if m < 40)
    
    return {
        "subject_name": matching_subject_name,
        "total_appeared": len(all_scores),
        "average_marks": avg_marks,
        "highest_marks": max_marks,
        "lowest_marks": min_marks,
        "subject_toppers": [{"usn": t["student_usn"], "name": t["student_name"], "marks": t["marks"]} for t in toppers],
        "pass_rate": round(((len(all_scores) - fail_count) / len(all_scores)) * 100, 2),
        "failures_count": fail_count
    }

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
            "higher": st1["name"] if sg1 > sg2 else (st2["name"] if sg2 > sg1 else "Tie"),
            "difference": abs(diff)
        })
        
    return {
        "student1": {"usn": st1["usn"], "name": st1["name"], "rank": st1["rank"], "cgpa": st1["cgpa"], "standing": st1["standing"]},
        "student2": {"usn": st2["usn"], "name": st2["name"], "rank": st2["rank"], "cgpa": st2["cgpa"], "standing": st2["standing"]},
        "cgpa_difference": round(abs(st1["cgpa"] - st2["cgpa"]), 2),
        "cgpa_winner": st1["name"] if st1["cgpa"] > st2["cgpa"] else (st2["name"] if st2["cgpa"] > st1["cgpa"] else "Tie"),
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
