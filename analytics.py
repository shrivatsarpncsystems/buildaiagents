import json
import os
import re
import difflib
from typing import Dict, Any, List, Optional, Tuple

DATA_FILE = os.path.join(os.path.dirname(__file__), "students_results.json")

WORD_TO_NUM = {
    "first": 1, "1st": 1, "one": 1, "1": 1,
    "second": 2, "2nd": 2, "two": 2, "2": 2,
    "third": 3, "3rd": 3, "three": 3, "3": 3,
    "fourth": 4, "4th": 4, "four": 4, "4": 4,
    "fifth": 5, "5th": 5, "five": 5, "5": 5,
    "sixth": 6, "6th": 6, "six": 6, "6": 6,
    "seventh": 7, "7th": 7, "seven": 7, "7": 7,
    "eighth": 8, "8th": 8, "eight": 8, "8": 8,
}

def extract_semester(text: str) -> Optional[int]:
    """
    Robust semester extractor supporting ordinals (3rd, third), words,
    misspellings (semister, semster), prefixes (sem 3, 3rd sem), and suffixes.
    """
    if not text:
        return None
    t = text.lower()
    
    # 1. Pattern: <sem_word> <num/ordinal/word> e.g. "sem 3", "semester 3rd", "semister third", "sem-2"
    m1 = re.search(r"\b(?:sem|semester|semister|semster|s)\s*[-:]?\s*([1-8](?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth)\b", t)
    if m1:
        val = m1.group(1).lower()
        if val in WORD_TO_NUM:
            return WORD_TO_NUM[val]
        num_str = re.sub(r"[^\d]", "", val)
        if num_str and 1 <= int(num_str) <= 8:
            return int(num_str)
            
    # 2. Pattern: <num/ordinal/word> <sem_word> e.g. "3rd semister", "third semester", "3 sem", "2nd sem"
    m2 = re.search(r"\b([1-8](?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth)\s*[-:]?\s*(?:sem|semester|semister|semster|sems)\b", t)
    if m2:
        val = m2.group(1).lower()
        if val in WORD_TO_NUM:
            return WORD_TO_NUM[val]
        num_str = re.sub(r"[^\d]", "", val)
        if num_str and 1 <= int(num_str) <= 8:
            return int(num_str)
            
    # 3. Direct mention with ordinal when in context of student/exam/top
    m3 = re.search(r"\b(?:in|of|for)?\s*([1-8](?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth)\b", t)
    if m3 and any(w in t for w in ["top", "topper", "toppers", "student", "students", "fail", "failed", "sgpa", "rank", "marks", "exam"]):
        val = m3.group(1).lower()
        if val in WORD_TO_NUM:
            return WORD_TO_NUM[val]
        num_str = re.sub(r"[^\d]", "", val)
        if num_str and 1 <= int(num_str) <= 8:
            return int(num_str)
            
    return None

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

def find_student(identifier: str, recent_usns: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Find a student by USN or name, prioritizing recently discussed candidates in conversation history."""
    students = _load_data()
    q = identifier.strip().lower()
    
    # 1. Exact USN match
    for s in students:
        if s["usn"].lower() == q:
            return s
            
    # 2. Prioritize recent students from conversation history (resolves typos like 'Aditya Verma' -> 'Aditi Verma')
    if recent_usns:
        st_map = {s["usn"]: s for s in students}
        q_tokens = [tok for tok in q.split() if len(tok) >= 3]
        for ru in recent_usns:
            if ru in st_map:
                cand = st_map[ru]
                cand_name = cand["name"].lower()
                cand_parts = cand_name.split()
                # Check token overlap
                if any(tok in cand_parts for tok in q_tokens):
                    return cand
                # Check fuzzy similarity
                ratio = difflib.SequenceMatcher(None, q, cand_name).ratio()
                if ratio >= 0.7:
                    return cand

    # 3. Exact, substring, or fuzzy name match
    matches = find_students_by_name(identifier)
    if matches:
        return matches[0]
        
    return None

def find_students_by_name(name: str) -> List[Dict[str, Any]]:
    """
    Find all students matching a full, partial, or typo-tolerant name
    (handles typos like 'divya bath' -> 'Divya Bhat', 'yash varma' -> 'Yash Verma').
    """
    students = _load_data()
    q = name.strip().lower()
    if not q or len(q) < 2:
        return []
        
    # 1. Exact full name match
    exact = [s for s in students if s["name"].lower() == q]
    if exact:
        return exact
        
    # 2. Substring match
    substr = [s for s in students if q in s["name"].lower()]
    if substr:
        return substr
        
    # 3. Fuzzy match full name using difflib
    name_map: Dict[str, List[Dict[str, Any]]] = {}
    for s in students:
        nl = s["name"].lower()
        if nl not in name_map:
            name_map[nl] = []
        name_map[nl].append(s)
        
    close = difflib.get_close_matches(q, name_map.keys(), n=2, cutoff=0.7)
    if close:
        results = []
        for c in close:
            results.extend(name_map[c])
        return results
        
    # 4. Token match with fallback (filtering out common conversational English words)
    STOP_WORDS = {
        "and", "the", "for", "with", "from", "can", "you", "give", "gie", "show", 
        "what", "tell", "score", "scores", "marks", "mark", "grade", "grades",
        "card", "cards", "scorecard", "scorecards",
        "sem", "semester", "semister", "semster", "sems", "student", "students", 
        "candidate", "candidates", "who", "which", "calculate", "calculatemarks",
        "percentage", "percent", "format", "formatt", "if", "ask", "particular",
        "every", "eery", "subject", "subjects", "individually", "indiidulally",
        "well", "total", "please", "how", "much", "did", "get", "got", "obtain",
        "obtained", "result", "results", "about", "also", "both", "all"
    }
    tokens = [tok for tok in q.split() if len(tok) >= 3 and tok not in STOP_WORDS]
    if tokens:
        matches = [s for s in students if any(tok in [part.lower() for part in s["name"].split()] for tok in tokens)]
        if matches:
            return matches
            
    return []

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
    # Discrete Mathematics (prioritized aliases including common typos like 'discreet')
    "discrete mathematical structures": ["Discrete Mathematical Structures"],
    "discrete mathematical": ["Discrete Mathematical Structures"],
    "discrete structures": ["Discrete Mathematical Structures"],
    "discrete maths": ["Discrete Mathematical Structures"],
    "discrete math": ["Discrete Mathematical Structures"],
    "discrete": ["Discrete Mathematical Structures"],
    "discreet mathematical structures": ["Discrete Mathematical Structures"],
    "discreet mathematical": ["Discrete Mathematical Structures"],
    "discreet structures": ["Discrete Mathematical Structures"],
    "discreet maths": ["Discrete Mathematical Structures"],
    "discreet math": ["Discrete Mathematical Structures"],
    "discreet": ["Discrete Mathematical Structures"],
    "dms": ["Discrete Mathematical Structures"],

    # Engineering Mathematics Domain (Semesters 1 & 2)
    "engineering mathematics": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "engineering maths": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "engineering math": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "engg mathematics": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "engg maths": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "engg math": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "eng maths": ["Engineering Mathematics I", "Engineering Mathematics II"],
    "eng math": ["Engineering Mathematics I", "Engineering Mathematics II"],

    # Engineering Mathematics I
    "engineering mathematics i": ["Engineering Mathematics I"],
    "engineering mathematics 1": ["Engineering Mathematics I"],
    "engineering maths 1": ["Engineering Mathematics I"],
    "engineering math 1": ["Engineering Mathematics I"],
    "math 1": ["Engineering Mathematics I"],
    "maths 1": ["Engineering Mathematics I"],
    "m1": ["Engineering Mathematics I"],

    # Engineering Mathematics II
    "engineering mathematics ii": ["Engineering Mathematics II"],
    "engineering mathematics 2": ["Engineering Mathematics II"],
    "engineering maths 2": ["Engineering Mathematics II"],
    "engineering math 2": ["Engineering Mathematics II"],
    "math 2": ["Engineering Mathematics II"],
    "maths 2": ["Engineering Mathematics II"],
    "m2": ["Engineering Mathematics II"],

    # Overall Mathematics Domain (All 3 Math Courses)
    "maths": ["Engineering Mathematics I", "Engineering Mathematics II", "Discrete Mathematical Structures"],
    "math": ["Engineering Mathematics I", "Engineering Mathematics II", "Discrete Mathematical Structures"],
    "mathematics": ["Engineering Mathematics I", "Engineering Mathematics II", "Discrete Mathematical Structures"],

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

def get_course_metadata(course_name: str) -> Dict[str, Any]:
    """Retrieve official course code, semester, and credits for a course name."""
    students = _load_data()
    for sm in range(1, 9):
        for sub in students[0]["semesters"][str(sm)]["subjects"]:
            if sub["name"].lower() == course_name.lower():
                return {
                    "name": sub["name"],
                    "code": sub["code"],
                    "semester": sm,
                    "credits": sub.get("credits", 4)
                }
    return {"name": course_name, "code": "N/A", "semester": "N/A", "credits": 4}

def detect_query_ambiguity(query_text: str, history=None) -> Optional[Dict[str, Any]]:
    """
    Detects if a user query is underspecified, ambiguous, or lacks crucial parameters
    (e.g., 'top scorer in maths', 'top 5 students in engineering maths', 'who topped machine learning',
    or a student name shared by multiple candidates).
    Returns an ambiguity descriptor dict with numbered clarification options, or None if query is clear.
    """
    if not query_text:
        return None
    q = query_text.lower().strip()
    
    # 0. Check if this query is already a clarification response to a previous counter-question
    if history and len(history) > 0:
        last_turn = history[-1]
        last_ans = last_turn.get("answer", "")
        if any(phrase in last_ans.lower() for phrase in [
            "which specific course", "which candidate's record", "which one would you like"
        ]):
            # This is an answer to a clarification question; do not treat as new ambiguity
            return None

    # 1. Multi-course domain check (e.g. 'maths', 'engineering maths', 'machine learning', 'web')
    sem = extract_semester(q)
    has_specific_sem = (sem is not None)
    has_specific_type = any(w in q for w in [
        "theory", "lab", "laboratory", "phase 1", "phase 2", "phase i", "phase ii",
        "m1", "m2", "math 1", "math 2", "maths 1", "maths 2", "dms"
    ])
    has_explicit_combined = any(w in q for w in [
        "overall", "combined", "both", "all courses", "all maths", "best at maths overall"
    ])
    
    # If user has NOT specified semester, course type, or explicit combined intent:
    if not has_specific_sem and not has_specific_type and not has_explicit_combined:
        for k in sorted(DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
            pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
            if re.search(pattern, q):
                courses = DOMAIN_SUBJECT_MAPPINGS[k]
                if len(courses) > 1:
                    # Multi-course domain found and underspecified!
                    course_meta = [get_course_metadata(c) for c in courses]
                    return {
                        "type": "multi_course_domain",
                        "domain_key": k,
                        "domain_title": k.title(),
                        "courses": course_meta,
                        "original_query": query_text
                    }
                    
    # 2. Duplicate candidate name check (e.g. 'show scorecard for Divya Bhat')
    if not any(w in q for w in ["top", "topper", "average", "fail", "placement", "compare", "grade", "distribution"]):
        clean_name = re.sub(r"\b(?:who|is|tell|me|about|show|profile|details|student|candidate|scorecard|score|card|transcript|record|of|for|please)\b", " ", q).strip()
        if len(clean_name) >= 3 and not re.search(r"1rv21cs\d{3}", q):
            exact_matches = find_students_by_name(clean_name)
            # Filter to candidates whose exact full name matches
            exact_cands = [s for s in exact_matches if s["name"].lower() == clean_name.lower()]
            if len(exact_cands) > 1:
                return {
                    "type": "duplicate_name",
                    "name": clean_name.title(),
                    "candidates": exact_cands,
                    "original_query": query_text
                }
                
    return None

def format_clarification_question(ambiguity_info: Dict[str, Any]) -> str:
    """Format an interactive, polite clarifying counter-question with clear numbered options."""
    a_type = ambiguity_info.get("type")
    
    if a_type == "multi_course_domain":
        dom_title = ambiguity_info.get("domain_title", "Subject")
        courses = ambiguity_info.get("courses", [])
        
        lines = [
            f"The '{dom_title}' curriculum covers {len(courses)} separate courses across semesters:",
            ""
        ]
        for idx, c in enumerate(courses, 1):
            sem_info = f"Semester {c['semester']}" if c['semester'] != 'N/A' else "Curriculum"
            code_info = f"Course Code: {c['code']}" if c['code'] != 'N/A' else ""
            paren = " | ".join(filter(None, [sem_info, code_info]))
            lines.append(f"{idx}. {c['name']} ({paren})")
            
        combined_idx = len(courses) + 1
        lines.append(f"{combined_idx}. Combined Overall Performance (Across all {dom_title} courses)")
        lines.append("")
        lines.append(f"Which specific course would you like to see (e.g., type '1', '2', or 'both')?")
        return "\n".join(lines)
        
    elif a_type == "duplicate_name":
        name = ambiguity_info.get("name", "Student")
        cands = ambiguity_info.get("candidates", [])
        lines = [
            f"There are {len(cands)} candidates named '{name}' in the B.Tech CSE cohort:",
            ""
        ]
        for idx, c in enumerate(cands, 1):
            lines.append(f"{idx}. {c['name']} (USN: {c['usn']} | Batch Rank: #{c['rank']} | CGPA: {c['cgpa']:.2f} | Standing: {c['standing']})")
        lines.append("")
        lines.append(f"Which candidate's record would you like to inspect (e.g., type '1', '2', or the USN)?")
        return "\n".join(lines)
        
    return "Could you please clarify your question with more specific details?"

def resolve_clarification_from_history(user_input: str, history=None) -> Optional[str]:
    """
    Checks if user_input is answering a clarifying counter-question asked in the last turn.
    Synthesizes the complete, unambiguous query with intent, subject/candidate, and graph options.
    """
    if not history or len(history) == 0 or not user_input:
        return None
        
    last_turn = history[-1]
    last_q = (last_turn.get("question") or "").strip()
    last_ans = (last_turn.get("answer") or "").strip()
    inp = user_input.lower().strip()
    
    # 1. Check if last_ans was asking for multi-course domain clarification
    if "which specific course would you like to see" in last_ans.lower():
        course_lines = re.findall(r"(\d+)\.\s+([^\(\n]+)\s+\(([^)]*)\)", last_ans)
        if not course_lines:
            return None
            
        is_opt1 = any(re.search(rf"\b{re.escape(w)}\b", inp) for w in ["1", "1st", "first", "sem 1", "semester 1", "one"])
        is_opt2 = any(re.search(rf"\b{re.escape(w)}\b", inp) for w in ["2", "2nd", "second", "sem 2", "semester 2", "two"])
        is_opt3 = any(re.search(rf"\b{re.escape(w)}\b", inp) for w in ["3", "3rd", "third", "sem 3", "semester 3", "three"])
        is_both = any(re.search(rf"\b{re.escape(w)}\b", inp) for w in [
            "both", "all", "combined", "overall", "together", "everything", str(len(course_lines) + 1)
        ])
        
        for c in course_lines:
            c_name = c[1].strip()
            if c_name.lower() in inp or (c[0] == "1" and "math 1" in inp) or (c[0] == "2" and "math 2" in inp):
                if c[0] == "1":
                    is_opt1 = True
                elif c[0] == "2":
                    is_opt2 = True
                elif c[0] == "3":
                    is_opt3 = True
                    
        is_graph_needed = any(w in (last_q + " " + user_input).lower() for w in ["graph", "chart", "plot", "bar"])
        graph_suffix = " give graph also" if is_graph_needed else ""
        
        num_m = re.search(r"(top\s*\d+|top scorer|topper|toppers|average|class average|marks|highest|lowest|grade distribution)", last_q, re.IGNORECASE)
        intent_prefix = num_m.group(1) if num_m else "toppers"
        
        if is_opt1 and len(course_lines) >= 1:
            target_name = course_lines[0][1].strip()
            return f"{intent_prefix} in {target_name}{graph_suffix}"
        elif is_opt2 and len(course_lines) >= 2:
            target_name = course_lines[1][1].strip()
            return f"{intent_prefix} in {target_name}{graph_suffix}"
        elif is_opt3 and len(course_lines) >= 3:
            target_name = course_lines[2][1].strip()
            return f"{intent_prefix} in {target_name}{graph_suffix}"
        elif is_both:
            return f"{intent_prefix} in combined overall {graph_suffix}"

    # 2. Check if last_ans was asking for duplicate student name clarification
    if "which candidate's record would you like to inspect" in last_ans.lower():
        usns = re.findall(r"1RV21CS\d{3}", last_ans, re.IGNORECASE)
        if not usns:
            return None
            
        direct_match = re.search(r"1rv21cs\d{3}", inp, re.IGNORECASE)
        target_usn = None
        if direct_match:
            target_usn = direct_match.group(0).upper()
        elif any(re.search(rf"\b{re.escape(w)}\b", inp) for w in ["1", "1st", "first", "one"]):
            target_usn = usns[0].upper()
        elif any(re.search(rf"\b{re.escape(w)}\b", inp) for w in ["2", "2nd", "second", "two"]) and len(usns) >= 2:
            target_usn = usns[1].upper()
            
        if target_usn:
            is_graph_needed = any(w in (last_q + " " + user_input).lower() for w in ["graph", "chart", "plot", "bar"])
            graph_suffix = " give graph also" if is_graph_needed else ""
            return f"show scorecard for student {target_usn}{graph_suffix}"
            
    return None

def get_domain_courses(query: str) -> Optional[Tuple[str, List[str]]]:
    """
    Checks if the query refers to a multi-course academic domain (e.g. 'engineering maths', 'web', 'machine learning').
    Returns (domain_title, list_of_course_names) if matched and has >1 courses, else None.
    """
    if not query:
        return None
    q = query.lower().strip()
    
    # If a specific semester is mentioned, it's a single course request, not domain-wide
    sem = extract_semester(q)
    if sem is not None and any(w in q for w in ["math", "maths", "mathematics", "m1", "m2", "dms"]):
        return None

    # Check explicit domain mappings (sorted by length descending)
    for k in sorted(DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
        if re.search(pattern, q):
            courses = DOMAIN_SUBJECT_MAPPINGS[k]
            if len(courses) == 1:
                return None  # Specific single course matched, not domain-wide
            return (k.title(), courses)
    return None

def resolve_subject_name(query: str) -> Optional[str]:
    """
    Resolves any user subject reference (including typos like 'discreet', contractions, and codes)
    to a canonical course name (e.g. 'discreet maths' -> 'Discrete Mathematical Structures',
    'ml' -> 'Machine Learning & Pattern Recognition', 'os' -> 'Operating Systems').
    """
    if not query:
        return None
    q = query.lower().strip()
    students = _load_data()
    
    # 0. Check explicit semester disambiguation for math
    sem = extract_semester(q)
    if sem == 1 and any(w in q for w in ["math", "maths", "mathematics", "m1"]):
        return "Engineering Mathematics I"
    elif sem == 2 and any(w in q for w in ["math", "maths", "mathematics", "m2"]):
        return "Engineering Mathematics II"
    elif sem == 3 and any(w in q for w in ["math", "maths", "mathematics", "dms"]):
        return "Discrete Mathematical Structures"

    # 1. Check domain mapping keys (sorted by length descending so specific aliases take priority over generic words)
    for k in sorted(DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
        if re.search(pattern, q):
            return DOMAIN_SUBJECT_MAPPINGS[k][0]
            
    # 2. Check direct course name or code match
    for sm in range(1, 9):
        for sub in students[0]["semesters"][str(sm)]["subjects"]:
            s_name = sub["name"]
            s_code = sub["code"]
            if s_name.lower() in q or s_code.lower() in q:
                return s_name
                
    # 3. Fuzzy match across all 48 course names
    words = [w for w in re.split(r"\W+", q) if len(w) >= 3 and w not in [
        "give", "show", "bar", "graph", "chart", "how", "many", "students",
        "passed", "passes", "fail", "failed", "who", "what", "grade", "distribution"
    ]]
    all_courses = []
    for sm in range(1, 9):
        for sub in students[0]["semesters"][str(sm)]["subjects"]:
            all_courses.append(sub["name"])
            
    for w in words:
        for cname in all_courses:
            c_words = [cw.lower() for cw in re.split(r"\W+", cname)]
            if any(difflib.SequenceMatcher(None, w, cw).ratio() >= 0.8 for cw in c_words):
                return cname
                
    return None

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
    
    # 0. Check explicit semester disambiguation for math
    sem = extract_semester(q_clean)
    if sem == 1 and any(w in q_clean for w in ["math", "maths", "mathematics", "m1"]):
        matched_subjects = ["Engineering Mathematics I"]
        domain_name = "Engineering Mathematics I"
    elif sem == 2 and any(w in q_clean for w in ["math", "maths", "mathematics", "m2"]):
        matched_subjects = ["Engineering Mathematics II"]
        domain_name = "Engineering Mathematics II"
    elif sem == 3 and any(w in q_clean for w in ["math", "maths", "mathematics", "dms"]):
        matched_subjects = ["Discrete Mathematical Structures"]
        domain_name = "Discrete Mathematical Structures"
        
    # 1. Match against domain mappings (sorted by length descending)
    if not matched_subjects:
        for k in sorted(DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
            pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
            if re.search(pattern, q_clean):
                matched_subjects = DOMAIN_SUBJECT_MAPPINGS[k]
                domain_name = k.title()
                break
            
    # 2. Match directly against any course name or code in the curriculum
    if not matched_subjects:
        for sm_idx in range(1, 9):
            for sub in students[0]["semesters"][str(sm_idx)]["subjects"]:
                if sub["name"].lower() in q_clean or sub["code"].lower() in q_clean:
                    matched_subjects = [sub["name"]]
                    domain_name = sub["name"]
                    break
            if matched_subjects:
                break
                
    # 3. Fuzzy match for typos in domain/course names (e.g. "machin learning", "operatng system")
    if not matched_subjects:
        words = q_clean.split()
        phrases = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)] + words
        for p in phrases:
            if len(p) >= 4:
                close_k = difflib.get_close_matches(p, DOMAIN_SUBJECT_MAPPINGS.keys(), n=1, cutoff=0.75)
                if close_k:
                    matched_subjects = DOMAIN_SUBJECT_MAPPINGS[close_k[0]]
                    domain_name = close_k[0].title()
                    break

    if not matched_subjects:
        return None
        
    out = []
    out.append(f"[OFFICIAL SUBJECT & DOMAIN ACADEMIC ANALYTICS: {domain_name.upper()}]")
    out.append(f"Curriculum Scope: {len(matched_subjects)} Course(s) across all 200 Candidates in CSE Cohort")
    out.append(f"Courses Evaluated: {', '.join(matched_subjects)}\n")
    
    # Check if user requested top N performers (e.g. 'top 10', 'top 5', 'toppers')
    num_m = re.search(r"(?:top|first)\s*(\d+)", q_clean)
    top_k = int(num_m.group(1)) if num_m else (10 if any(w in q_clean for w in ["top", "topper", "toppers", "rank"]) else 5)
    top_k = max(3, min(top_k, 25))

    # 1. Course-by-Course Detailed Performance and Toppers FIRST
    out.append("COURSE-BY-COURSE DETAILED PERFORMANCE & TOPPERS:")
    for sub_idx, sub_name in enumerate(matched_subjects, 1):
        all_scores = []
        sub_code = ""
        sub_sem = ""
        for s in students:
            for sm_idx, sm in s["semesters"].items():
                for sub in sm["subjects"]:
                    if sub["name"] == sub_name:
                        sub_code = sub["code"]
                        sub_sem = sm_idx
                        all_scores.append({
                            "usn": s["usn"],
                            "name": s["name"],
                            "marks": sub["marks"],
                            "grade": sub["grade"],
                            "cgpa": s["cgpa"],
                            "standing": s["standing"],
                            "backlogs": s["backlogs"]
                        })
                        
        if not all_scores:
            continue
            
        all_scores.sort(key=lambda x: (x["marks"], x["cgpa"]), reverse=True)
        marks_vals = [x["marks"] for x in all_scores]
        avg_score = round(sum(marks_vals) / len(marks_vals), 2)
        max_score = marks_vals[0]
        min_score = marks_vals[-1]
        toppers = [x for x in all_scores if x["marks"] == max_score]
        fail_cnt = sum(1 for m in marks_vals if m < 40)
        pass_cnt = len(marks_vals) - fail_cnt
        pass_pct = round((pass_cnt / len(marks_vals)) * 100, 2)
        
        grade_counts = {g: sum(1 for x in all_scores if x["grade"] == g) for g in ["O", "A+", "A", "B+", "B", "C", "F"]}
        grade_str = ", ".join([f"Grade {g}: {c}" for g, c in grade_counts.items() if c > 0])
        
        top_str = ", ".join([f"{t['name']} (USN: {t['usn']})" for t in toppers[:3]])
        out.append(f"\nCourse {sub_idx}: {sub_name} (Code: {sub_code}, Semester {sub_sem})")
        out.append(f"  * Total Candidates Appeared: {len(marks_vals)} Students")
        out.append(f"  * Students Passed: {pass_cnt} Students ({pass_pct}%)")
        out.append(f"  * Students Failed (Backlogs): {fail_cnt} Students ({100-pass_pct:.1f}%)")
        out.append(f"  * Grade Breakdown: {grade_str}")
        out.append(f"  * Class Average: {avg_score}/100 | Lowest Score: {min_score}/100")
        out.append(f"  * Highest Score Achieved: {max_score}/100 by: {top_str}")
        
        out.append(f"  * Top {top_k} Performers in {sub_name} (Strictly Ranked by Course Marks in Descending Order):")
        for idx, st in enumerate(all_scores[:top_k], 1):
            out.append(f"    {idx}. {st['name']} (USN: {st['usn']}) - Marks: {st['marks']}/100 ({st['grade']}) | Cumulative CGPA: {st['cgpa']:.2f}")

    # 2. Combined Domain Merit Roll (when domain has multiple courses, e.g. Mathematics, AI, Labs)
    if len(matched_subjects) > 1:
        student_aggs = []
        for s in students:
            scores = []
            for sm in s["semesters"].values():
                for sub in sm["subjects"]:
                    if sub["name"] in matched_subjects:
                        scores.append({"name": sub["name"], "marks": sub["marks"], "grade": sub["grade"]})
            if len(scores) == len(matched_subjects):
                avg_m = round(sum(x["marks"] for x in scores) / len(scores), 2)
                student_aggs.append({"usn": s["usn"], "name": s["name"], "avg": avg_m, "scores": scores, "cgpa": s["cgpa"]})
                
        student_aggs.sort(key=lambda x: (x["avg"], x["cgpa"]), reverse=True)
        
        out.append(f"\nCOMBINED MERIT ROLL: {domain_name.upper()} (Combined Average across all {len(matched_subjects)} courses):")
        for idx, st in enumerate(student_aggs[:top_k], 1):
            sc_details = ", ".join([f"{sc['name']}: {sc['marks']}/100" for sc in st["scores"]])
            out.append(f"  {idx}. {st['name']} (USN: {st['usn']}) - Combined Average: {st['avg']}% | Cumulative CGPA: {st['cgpa']:.2f} [{sc_details}]")
        out.append("")
        out.append(
            "[CRITICAL RANKING AND ACCURACY RULES FOR LLM:\n"
            "- Individual course rankings MUST follow that course's marks in strictly descending order.\n"
            "- A student with lower marks MUST NEVER be placed above a student with higher marks.\n"
            "- Sneha Menon scored 85 in Engineering Mathematics I and is NOT in the Math I top 5.\n"
            "- If the user asks for Engineering Mathematics without specifying Semester 1 or Semester 2, clearly present BOTH Semester 1 (Math I) and Semester 2 (Math II) Top Performers, followed by the Combined Merit list.\n"
            "- DO NOT mislabel combined average percentages as single-course marks.]"
        )
        
    return "\n".join(out)

def get_subject_top_scorers(subject_query: str, n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Retrieve top N students in a specific course with their marks, grade, and cumulative CGPA.
    """
    target_subject = resolve_subject_name(subject_query)
    if not target_subject:
        return None
        
    students = _load_data()
    all_scores = []
    sub_code = ""
    sub_sem = ""
    for s in students:
        for sm_idx, sm in s["semesters"].items():
            for sub in sm["subjects"]:
                if sub["name"] == target_subject:
                    sub_code = sub["code"]
                    sub_sem = sm_idx
                    all_scores.append({
                        "usn": s["usn"],
                        "name": s["name"],
                        "marks": sub["marks"],
                        "grade": sub["grade"],
                        "cgpa": s["cgpa"],
                        "standing": s["standing"]
                    })
    if not all_scores:
        return None
        
    all_scores.sort(key=lambda x: (x["marks"], x["cgpa"]), reverse=True)
    n_clamped = max(1, min(n, 30))
    top_students = all_scores[:n_clamped]
    
    return {
        "subject_name": target_subject,
        "subject_code": sub_code,
        "semester": sub_sem,
        "n": len(top_students),
        "students": top_students
    }

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

def get_student_semester_breakdown(identifier: str, semester: int) -> str:
    """
    Detailed semester transcript breakdown with individual course marks,
    individual percentages, total marks obtained, and overall semester percentage.
    """
    st = find_student(identifier)
    if not st:
        return f"Student '{identifier}' not found in college examination records."
    sem_str = str(semester)
    if sem_str not in st["semesters"]:
        return f"Semester {semester} not found for student {st['name']} ({st['usn']})."
    sem = st["semesters"][sem_str]
    subs = sem["subjects"]
    
    total_obtained = sum(sub["marks"] for sub in subs)
    total_max = len(subs) * 100
    sem_percentage = round((total_obtained / total_max) * 100, 2)
    has_backlogs = any(sub["grade"] == "F" for sub in subs)
    
    out = [
        f"[OFFICIAL SEMESTER {semester} TRANSCRIPT & MARKS AUDIT: {st['name']} ({st['usn']})]",
        f"Program: B.Tech Computer Science & Engineering | Cohort Rank: #{st['rank']} of 200 Candidates",
        f"Semester {semester} SGPA: {sem['sgpa']:.2f} / 10.0 | Cumulative CGPA: {st['cgpa']:.2f} / 10.0",
        f"Credits Registered & Earned: {sem['credits']:.1f} Credits",
        f"Semester Result Status: {'ACTIVE ARREARS / BACKLOG' if has_backlogs else 'ALL COURSES CLEARED'}\n",
        f"SEMESTER MARKS SUMMARY & AGGREGATE PERCENTAGE:",
        f"• Total Marks Obtained: {total_obtained} / {total_max}",
        f"• Overall Semester Percentage: {sem_percentage:.2f}%",
        f"• Average Subject Score: {sem_percentage:.2f} / 100\n",
        "INDIVIDUAL COURSE MARKS & PERCENTAGE BREAKDOWN:"
    ]
    for sub in subs:
        sub_pct = (sub["marks"] / 100.0) * 100.0
        status = "PASS" if sub["grade"] != "F" else "FAIL (ARREAR)"
        out.append(
            f"- {sub['code']} {sub['name']} ({sub['credits']} Credits):\n"
            f"  * Marks Obtained: {sub['marks']} / 100 ({sub_pct:.1f}%)\n"
            f"  * Grade Awarded : Grade {sub['grade']} (Grade Point: {sub['grade_point']}) | Status: {status}"
        )
    return "\n".join(out)

def get_placement_eligibility(min_cgpa: float = 7.0, max_backlogs: int = 0, limit: int = 25) -> str:
    """Filter students meeting corporate campus placement drive cutoff criteria."""
    students = _load_data()
    eligible = [s for s in students if s["cgpa"] >= min_cgpa and s["backlogs"] <= max_backlogs]
    eligible.sort(key=lambda x: x["cgpa"], reverse=True)
    out = [
        f"[OFFICIAL TRAINING & PLACEMENT (TPO) ELIGIBILITY SCREENING]",
        f"Drive Screening Cutoff: Minimum CGPA >= {min_cgpa:.2f} | Maximum Active Backlogs <= {max_backlogs}",
        f"Total Eligible Candidates: {len(eligible)} of {len(students)} Students ({len(eligible)/len(students)*100:.1f}% of CSE Cohort)\n",
        f"Top Shortlisted Candidates (Showing top {min(limit, len(eligible))}):"
    ]
    for idx, s in enumerate(eligible[:limit], 1):
        out.append(f"{idx}. {s['name']} (USN: {s['usn']}) - CGPA: {s['cgpa']:.2f} | Rank: #{s['rank']} | Backlogs: {s['backlogs']} | Standing: {s['standing']}")
    return "\n".join(out)

def get_subject_grade_distribution(subject_query: str) -> Optional[str]:
    """Calculate exact grade distribution (O, A+, A, B+, B, C, F) for a subject."""
    students = _load_data()
    q = subject_query.lower().strip()
    
    target_name = None
    # 1. Match against domain mappings (e.g. ml, os, dbms, cn, maths)
    for k in sorted(DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
        if re.search(pattern, q):
            target_name = DOMAIN_SUBJECT_MAPPINGS[k][0]
            break
            
    # 2. Match directly against curriculum courses or codes
    if not target_name:
        for sm in range(1, 9):
            for sub in students[0]["semesters"][str(sm)]["subjects"]:
                if sub["name"].lower() in q or sub["code"].lower() in q:
                    target_name = sub["name"]
                    break
            if target_name:
                break
                
    if not target_name:
        return None
    all_grades = []
    all_marks = []
    for s in students:
        for sm in s["semesters"].values():
            for sub in sm["subjects"]:
                if sub["name"] == target_name:
                    all_grades.append(sub["grade"])
                    all_marks.append(sub["marks"])
                    
    total = len(all_grades)
    if total == 0:
        return None
        
    counts = {g: all_grades.count(g) for g in ["O", "A+", "A", "B+", "B", "C", "F"]}
    out = [
        f"[OFFICIAL GRADE DISTRIBUTION: {target_name.upper()}]",
        f"Cohort Enrollment: {total} Students | Course Average Score: {sum(all_marks)/total:.2f}/100 | Highest: {max(all_marks)}/100 | Lowest: {min(all_marks)}/100\n",
        "Grade Breakdown:",
        f"- Grade O  (90-100%): {counts['O']:3d} students ({counts['O']/total*100:.1f}%) [Outstanding]",
        f"- Grade A+ (80-89%) : {counts['A+']:3d} students ({counts['A+']/total*100:.1f}%) [Excellent]",
        f"- Grade A  (70-79%) : {counts['A']:3d} students ({counts['A']/total*100:.1f}%) [Very Good]",
        f"- Grade B+ (60-69%) : {counts['B+']:3d} students ({counts['B+']/total*100:.1f}%) [Good]",
        f"- Grade B  (50-59%) : {counts['B']:3d} students ({counts['B']/total*100:.1f}%) [Above Average]",
        f"- Grade C  (40-49%) : {counts['C']:3d} students ({counts['C']/total*100:.1f}%) [Pass]",
        f"- Grade F  (< 40%)  : {counts['F']:3d} students ({counts['F']/total*100:.1f}%) [Fail / Re-exam]",
        f"\nDistinction Honor Rate (Grade O & A+): {(counts['O'] + counts['A+'])/total*100:.1f}%",
        f"Course Pass Rate: {(total - counts['F'])/total*100:.1f}% ({counts['F']} backlog candidates)"
    ]
    return "\n".join(out)

def check_graduation_status(identifier: str) -> str:
    """Audit degree clearance and convocation eligibility for a candidate."""
    students = _load_data()
    st = find_student(identifier)
    if not st:
        return f"Student '{identifier}' not found in college examination records."
    cleared = (st["backlogs"] == 0 and st["total_credits"] >= 136.5)
    out = [
        f"[OFFICIAL DEGREE CLEARANCE & CONVOCATION AUDIT: {st['name']} ({st['usn']})]",
        f"Program: B.Tech Computer Science & Engineering (Batch 2021-2025 | Autonomous Gazette)",
        f"Cohort Rank: #{st['rank']} of 200 Candidates | Final Cumulative CGPA: {st['cgpa']:.2f} / 10.0",
        f"Cumulative Credits Earned: {st['total_credits']:.1f} / 136.5 Total Program Credits",
        f"Active Backlogs: {st['backlogs']}",
        f"Graduating Academic Standing: {st['standing']}",
        f"Graduation Audit Decision: {'CLEARED FOR DEGREE CONFERMENT (CONVOCATION READY)' if cleared else 'PENDING - REMEDIATION REQUIRED (ACTIVE ARREARS)'}"
    ]
    return "\n".join(out)
