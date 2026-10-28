import os
import re
import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, retrieve_with_sources
import analytics
import visualizer

from typing import Optional

# 1. Initialize local LLM with zero temperature for deterministic, strictly grounded responses
model = OllamaLLM(model="llama3.2:3b", temperature=0.0)

# 2. Strict RAG Prompt Template with Grounding, Multi-turn History (k >= 6), Citations & Guardrails
template = """You are the Official Academic AI Assistant for Apex Institute of Technology (Department of Computer Science & Engineering).
Your role is to assist students, faculty, and administrators with examination records, transcripts, SGPA/CGPA calculations, merit lists, and academic regulations.

==================================================
ACADEMIC CONTEXT & DATA (OFFICIAL DATABASE TRUTH):
{context}

SOURCE CITATIONS AVAILABLE:
{citations}
==================================================

==================================================
PREVIOUS CONVERSATION HISTORY (FOR CONTEXT & PRONOUN RESOLUTION ONLY):
{chat_history}
==================================================

USER QUESTION: {question}

STRICT INSTRUCTIONS FOR YOUR RESPONSE:
1. ABSOLUTELY NO META-TALK: NEVER start responses with "Based on the provided conversation history", "Based on the academic context", "Based on available information", "Please note that the conversation history...", or conversational disclaimers. Start DIRECTLY with the answer.
2. OFFICIAL CALCULATIONS PRECEDENCE: The ACADEMIC CONTEXT contains exact, verified calculations across all 200 students. You MUST use those exact details, marks, ranks, and numbers.
3. CONVERSATION CONTEXT & PRONOUN RESOLUTION: Use the PREVIOUS CONVERSATION HISTORY solely to resolve pronouns (such as "they", "their", "thei", "them", "both", "he", "she", "these students", "the two students") or follow-up questions referencing students, subjects, or semesters previously discussed.
4. DIRECT & ACCURATE ANSWER FORMATTING:
   - For student comparisons (e.g. "Compare student 1RV21CS021 and 1RV21CS008"):
     You MUST provide a comprehensive, side-by-side comparative analysis:
     a. Candidate 1 Profile: Full Name, USN, Batch Rank, Cumulative CGPA, Academic Standing, Backlogs, Best/Lowest Semester.
     b. Candidate 2 Profile: Full Name, USN, Batch Rank, Cumulative CGPA, Academic Standing, Backlogs, Best/Lowest Semester.
     c. Comparative Summary: Exact CGPA difference, Higher CGPA candidate, Higher Rank candidate, and a side-by-side comparison of their 8-semester SGPA progression. NEVER provide a single-line answer.
   - For student scorecards & transcripts (e.g. "give me Aditi Verma score card", "scorecard for 1RV21CS021"):
     You MUST provide the full academic record:
     a. Student Profile: Full Name, USN, Batch Rank, Cumulative CGPA, Academic Standing, Backlogs, Total Credits, and 8-Semester Marks Average.
     b. Key Milestones: Best Semester, Lowest Semester, Highest Scoring Course, Lowest Scoring Course.
     c. Semester Progression: List each semester's SGPA and course-by-course marks obtained out of 100 with grades.
   - For semester toppers (e.g. "top 10 students in 3rd semester"), present the ranked list strictly by their Semester SGPA, showing Rank, Name, USN, and SGPA.
   - For failed students or their marks, ALWAYS list the student name, USN, semester SGPA, and each failed course along with their exact marks obtained out of 100 and Grade F (e.g. "Deepak Reddy (1RV21CS180) - Engineering Chemistry: 30/100 (Grade: F)").
   - For individual student semester marks / scores / percentages:
     You MUST output ALL of the following:
     a. Candidate Header: Student Name, USN, and Semester SGPA.
     b. Individual Course Breakdown: List EVERY course individually with its name, marks obtained out of 100, individual percentage (e.g. 93.0%), and grade awarded.
     c. Total Marks & Aggregate Percentage: Clearly state Total Marks Obtained out of maximum marks (e.g. 562 / 600) and the Overall Semester Percentage (e.g. 93.67%).
   - For questions asking "how many students passed/failed in [subject]" or course pass rates (e.g. "how many students passes in discreet maths"):
     Provide a concise 3-4 line statistical executive summary: Total Candidates, Number & Percentage of Students Passed, Number & Percentage with Backlogs (Failed), and Class Average.
     CRITICAL: NEVER print an exhaustive list, directory, or score table of all individual passing students. Keep it strictly to the aggregate summary numbers.
   - For graph or visualization queries (e.g. "give bar graph of...", "give graph for previous conversation", "plot comparison"):
     Keep the text response brief and to the point (under 4 lines) so the conversation moves forward immediately without delays. Announce the visual chart rendered and summarize the key metrics.
   - For subject top performers or rankings (e.g. "top 5 students in engineering maths", "toppers in ML"):
     a. Individual Course Rankings: For any single course (e.g. Engineering Mathematics I or II), students MUST be listed in strictly descending order of their marks obtained in that course (e.g. 99, 99, 95, 95, 94). NEVER place a student with lower marks above a student with higher marks.
     b. Multi-Course Domains (e.g. "Engineering Maths"): Present BOTH Semester 1 (Engineering Mathematics I) and Semester 2 (Engineering Mathematics II) course toppers separately, followed by the Combined Merit list.
     c. Strict Integrity: NEVER mislabel a combined multi-course average as an individual course score. Only report verified marks directly present in the database calculations.
   - If multiple candidates share the same name (e.g. Divya Bhat), always distinguish them explicitly by USN and rank.
5. STRICT GROUNDING & REFUSAL: Do NOT invent student marks, ranks, or policies. If the question cannot be answered from the provided records or conversation history (e.g. hostel fees, sports, cafeteria menu), explicitly state:
   "I do not have supporting context in the official college examination records to answer this question."
6. ANTI-AMBIGUITY & CONSISTENCY CHECK:
   - Always compare what the user asked with what data is present in the context before answering.
   - If user asks for an ambiguous course or ranking without clarification, ask a counter-question instead of guessing.
   - For course rankings, marks MUST be strictly monotonic in descending order (Rank 1 >= Rank 2 >= Rank 3). Never invert student marks or ranks.
7. CITATIONS: Include the source citation at the end of your answer (e.g. "[Source: students_results.json / Autonomous Examination Database]").

OFFICIAL RESPONSE:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

MAX_HISTORY_TURNS = 6  # Maintains at least k=6 turns of conversation memory

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

def format_chat_history(history, k=MAX_HISTORY_TURNS) -> str:
    """Format the last k turns of conversation into readable dialog history."""
    if not history:
        return "No previous conversation history. This is the start of the session."
    recent = history[-k:]
    lines = []
    for idx, turn in enumerate(recent, 1):
        lines.append(f"Turn {idx}:")
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
        lines.append("")
    return "\n".join(lines).strip()

def extract_recent_entities(history, max_turns=MAX_HISTORY_TURNS):
    """
    Extracts recently discussed student USNs, semesters, and last turn intent from history.
    Returns entities ordered from most recent to oldest.
    """
    recent_usns = []
    recent_sems = []
    last_turn_intent = None
    
    if not history:
        return {"usns": [], "semesters": [], "last_turn_intent": None}
        
    last_turn = history[-1]
    last_q = (last_turn.get("question") or last_turn.get("query") or "").lower()
    if any(w in last_q for w in ["fail", "failed", "backlog", "backlogs", "arrear", "arrears"]):
        last_turn_intent = "failed_students"
    elif any(w in last_q for w in ["compare", "versus", "vs", "difference"]):
        last_turn_intent = "comparison"
    elif any(w in last_q for w in ["top", "topper", "toppers", "rank"]):
        last_turn_intent = "toppers"
    elif any(w in last_q for w in ["placement", "drive", "shortlist"]):
        last_turn_intent = "placement"
        
    for turn in reversed(history[-max_turns:]):
        q_text = turn.get("question") or turn.get("query") or ""
        a_text = turn.get("answer") or turn.get("response") or ""
        text = f"{q_text} {a_text}"
        
        # Extract USNs (e.g. 1RV21CS028)
        found_usns = re.findall(r"1rv21cs\d{3}", text, re.IGNORECASE)
        for u in found_usns:
            u_upper = u.upper()
            if u_upper not in recent_usns:
                recent_usns.append(u_upper)
                
        # Extract Semesters using robust extractor
        q_sem = extract_semester(q_text)
        if q_sem and q_sem not in recent_sems:
            recent_sems.append(q_sem)
        a_sems = re.findall(r"(?:sem|semester)\s*([1-8])", text, re.IGNORECASE)
        for sm in a_sems:
            sm_int = int(sm)
            if sm_int not in recent_sems:
                recent_sems.append(sm_int)
                
    return {
        "usns": recent_usns,
        "semesters": recent_sems,
        "last_turn_intent": last_turn_intent
    }

def route_and_build_context(user_query: str, history=None, show_graph: bool = True):
    """
    Intelligent Query Router:
    Combines deterministic analytics calculation engine with Chroma vector search.
    Resolves conversational pronouns and references across multi-turn dialogs (k >= 6).
    Dispatches interactive visual charts (matplotlib, seaborn, pandas) when requested.
    """
    if history is None:
        history = []
        
    q_lower = user_query.lower().strip()
    additional_data = ""
    citations_list = []

    recent_entities = extract_recent_entities(history, max_turns=MAX_HISTORY_TURNS)
    recent_usns = recent_entities["usns"]
    recent_sems = recent_entities["semesters"]
    last_intent = recent_entities.get("last_turn_intent")

    # 0. Check for Academic Regulations & Grading Scheme Policy
    is_policy_query = any(w in q_lower for w in [
        "grading", "grade policy", "grading policy", "criteria for grade",
        "regulation", "regulations", "grading scheme", "passing criteria",
        "award criteria", "grade o", "grade a+", "letter grade"
    ])
    if is_policy_query:
        from vector import vector_store
        p1 = vector_store.get(where={"page": 1})
        if p1 and p1["documents"]:
            additional_data += f"\n[OFFICIAL ACADEMIC REGULATION & GRADING POLICY]:\n{p1['documents'][0]}\n"
            citations_list.append("[Source: college_exam_results.pdf, Page 1]")

    # Detect USNs directly in user question
    direct_usns = [u.upper() for u in list(dict.fromkeys(re.findall(r"1rv21cs\d{3}", q_lower)))]

    # Pronoun / follow-up reference detection
    ref_pronouns = [
        "they", "their", "them", "both", "these", "those",
        "he", "she", "his", "her", "the two", "the students", "respective",
        "names", "former", "latter", "first one", "second one"
    ]
    has_pronoun_reference = any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in ref_pronouns) or "thei respectie" in q_lower or "their respectie" in q_lower
    has_followup_reference = has_pronoun_reference or any(w in q_lower for w in ["previous", "prior", "last", "for this", "for that", "earlier", "above", "them", "both"])
    is_compare_intent = any(w in q_lower for w in [
        "compare", "comparison", "difference", "versus", "vs", "between",
        "who scored higher", "who has higher", "who is better", "which one", "higher cgpa"
    ])
    is_marks_followup = any(w in q_lower for w in ["mark", "marks", "score", "scores", "grade", "grades"]) and (
        has_followup_reference or any(w in q_lower for w in ["give", "show", "what", "their", "thei", "respectie", "respective"])
    )

    # 1. Comparison of Two Students (Direct USNs or Follow-up referencing recent students)
    target_compare_usns = []
    if len(direct_usns) >= 2:
        target_compare_usns = direct_usns[:2]
    elif is_compare_intent and len(direct_usns) == 1 and len(recent_usns) >= 1:
        target_compare_usns = [recent_usns[0], direct_usns[0]]
    elif (is_compare_intent or (has_followup_reference and (last_intent == "comparison" or len(recent_usns) >= 2))) and len(direct_usns) == 0 and len(recent_usns) >= 2:
        target_compare_usns = recent_usns[:2]

    if target_compare_usns:
        u1, u2 = target_compare_usns[0], target_compare_usns[1]
        comp_res = analytics.compare_students(u1, u2)
        st1 = analytics.find_student(u1)
        st2 = analytics.find_student(u2)
        
        if st1 and st2:
            sc1 = analytics.get_student_scorecard(u1)
            sc2 = analytics.get_student_scorecard(u2)
            
            same_name_note = ""
            if st1["name"].lower() == st2["name"].lower():
                same_name_note = (
                    f"CRITICAL DISAMBIGUATION NOTICE (IDENTICAL NAMES):\n"
                    f"Both candidates have the EXACT SAME FULL NAME: '{st1['name']}'.\n"
                    f"They are two separate individual students registered in the 200-student CSE batch.\n"
                    f"- Candidate 1: {st1['name']} (USN: {st1['usn']}, Batch Rank: #{st1['rank']}, CGPA: {st1['cgpa']:.2f})\n"
                    f"- Candidate 2: {st2['name']} (USN: {st2['usn']}, Batch Rank: #{st2['rank']}, CGPA: {st2['cgpa']:.2f})\n"
                )
            
            additional_data += (
                f"\n[OFFICIAL SIDE-BY-SIDE CANDIDATE COMPARISON & RECORD]:\n"
                f"{same_name_note}\n"
                f"Candidate 1 Academic Record:\n"
                f"  - Official Full Name: {st1['name']}\n"
                f"  - University Seat Number (USN): {st1['usn']}\n"
                f"  - Class Rank: #{st1['rank']} of 200 Students\n"
                f"  - Cumulative CGPA: {st1['cgpa']:.2f} / 10.0\n"
                f"  - Academic Standing: {st1['standing']}\n"
                f"  - Active Backlogs: {st1['backlogs']}\n"
                f"  - Total Credits: {st1['total_credits']}\n"
                f"  - 8-Semester Marks Average: {sc1.get('average_marks', 'N/A')}/100\n"
                f"  - Best Semester: {sc1.get('best_semester', 'N/A')} | Lowest: {sc1.get('lowest_semester', 'N/A')}\n"
                f"  - Semester SGPA Progression: {json.dumps(sc1.get('sgpa_by_semester', {}))}\n"
                f"\n"
                f"Candidate 2 Academic Record:\n"
                f"  - Official Full Name: {st2['name']}\n"
                f"  - University Seat Number (USN): {st2['usn']}\n"
                f"  - Class Rank: #{st2['rank']} of 200 Students\n"
                f"  - Cumulative CGPA: {st2['cgpa']:.2f} / 10.0\n"
                f"  - Academic Standing: {st2['standing']}\n"
                f"  - Active Backlogs: {st2['backlogs']}\n"
                f"  - Total Credits: {st2['total_credits']}\n"
                f"  - 8-Semester Marks Average: {sc2.get('average_marks', 'N/A')}/100\n"
                f"  - Best Semester: {sc2.get('best_semester', 'N/A')} | Lowest: {sc2.get('lowest_semester', 'N/A')}\n"
                f"  - Semester SGPA Progression: {json.dumps(sc2.get('sgpa_by_semester', {}))}\n"
                f"\n"
                f"Direct Comparison Summary:\n"
                f"  - CGPA Difference: {comp_res.get('cgpa_difference', 0.0):.2f}\n"
                f"  - Higher Cumulative CGPA: {comp_res.get('cgpa_winner', 'Tie')}\n"
                f"  - Higher Batch Rank: #{min(st1['rank'], st2['rank'])} ({st1['name'] if st1['rank'] < st2['rank'] else st2['name']} - USN: {st1['usn'] if st1['rank'] < st2['rank'] else st2['usn']})\n"
            )
            citations_list.append("[Source: college_exam_results.pdf, Page 1]")
            citations_list.append("[Source: students_results.json / Autonomous Examination Database]")

    # 2. Check for Specific Student Semester Detailed Transcript & Score/Percentage
    sem_from_query = extract_semester(q_lower)
    target_st_id = None
    if direct_usns:
        target_st_id = direct_usns[0]
    else:
        # Check if user query mentions a student name
        clean_name = re.sub(r"\b(?:sem|semester|semister|semster|s)\s*[-:]?\s*[1-8](?:st|nd|rd|th)?\b", "", q_lower)
        clean_name = re.sub(r"\b[1-8](?:st|nd|rd|th)?\s*(?:sem|semester|semister|semster|sems)\b", "", clean_name)
        clean_name = re.sub(r"\b(?:score|scores|marks|mark|percentage|percent|result|results|grade|grades|of|in|for|student|candidate|show|what|calculate|give|graph|plot|chart|visualize|bar)\b", " ", clean_name).strip()
        if clean_name and len(clean_name) >= 3:
            matched_students = analytics.find_students_by_name(clean_name)
            if matched_students:
                target_st_id = matched_students[0]["usn"]
        if not target_st_id and (has_followup_reference and recent_usns):
            target_st_id = recent_usns[0]

    effective_sem = sem_from_query or (recent_sems[0] if (has_followup_reference and recent_sems) else None)
    if effective_sem and target_st_id and not target_compare_usns:
        additional_data += f"\n{analytics.get_student_semester_breakdown(target_st_id, effective_sem)}\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf]")

    # 3. Check for Follow-Up Question on Marks for Previously Discussed Failed Students
    elif is_marks_followup and last_intent == "failed_students" and recent_sems:
        active_sem = recent_sems[0]
        fail_data = analytics.get_failed_students(semester=active_sem)
        additional_data += f"\n[OFFICIAL CANDIDATES WHO FAILED IN SEMESTER {active_sem} - EXACT SUBJECT MARKS & GRADES]:\n"
        for st in fail_data["students"]:
            additional_data += f"- {st['name']} (USN: {st['usn']}) | Semester {active_sem} SGPA: {st['sgpa']:.2f} | Failed Course(s): {', '.join(st['failed_subjects'])}\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf]")

    # 4. Check for Degree Clearance & Graduation Convocation Audit
    elif any(w in q_lower for w in ["graduation", "graduate", "degree clearance", "eligible to graduate", "degree conferment", "convocation", "degree audit"]):
        target_usn = direct_usns[0] if direct_usns else (recent_usns[0] if recent_usns else None)
        if target_usn:
            additional_data += f"\n{analytics.check_graduation_status(target_usn)}\n"
        else:
            students = analytics._load_data()
            cleared = sum(1 for s in students if s["backlogs"] == 0 and s["total_credits"] >= 136.5)
            additional_data += (
                f"\n[OFFICIAL COHORT GRADUATION CLEARANCE AUDIT (200 STUDENTS)]:\n"
                f"- Total Candidates Evaluated: 200 Students\n"
                f"- Cleared for Degree Conferment: {cleared} Students ({cleared/200*100:.1f}%)\n"
                f"- Remediation Required (Active Backlogs): {200 - cleared} Students\n"
            )
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf, Page 1]")

    # 5. Check for Corporate Campus Placement & TPO Eligibility Screening (e.g. "placement", "campus drive", "cutoff")
    elif any(w in q_lower for w in ["placement", "placements", "campus drive", "tier 1", "tier-1", "placement drive", "eligible for drive", "job eligibility", "cutoff", "shortlist", "eligible for placement"]):
        if "tier 1" in q_lower or "tier-1" in q_lower:
            cutoff = 8.5
        else:
            num_match = re.search(r"(?:cgpa|cutoff|above|>=|>)\s*(\d+(?:\.\d+)?)", q_lower)
            cutoff = float(num_match.group(1)) if num_match else 7.0
        additional_data += f"\n{analytics.get_placement_eligibility(min_cgpa=cutoff, max_backlogs=0)}\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf, Page 1]")

    # 6. Check for Subject Grade Distribution (e.g. "Grade distribution in ML", "How many students got Grade O in OS")
    elif any(w in q_lower for w in ["grade distribution", "how many students got grade", "grade breakdown", "distribution of grades", "grades count", "how many grade o", "how many grade a"]):
        dist_res = analytics.get_subject_grade_distribution(q_lower)
        if dist_res:
            additional_data += f"\n{dist_res}\n"
            citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
            citations_list.append("[Source: college_exam_results.pdf]")

    # 7. Single Student Scorecard Lookup (Direct USN, Name search, or Follow-up pronoun)
    if not additional_data:
        is_scorecard_query = any(w in q_lower for w in ["scorecard", "score card", "profile", "record", "all semester", "all sem", "marks average", "performance"])
        target_scorecard_usn = None
        
        if direct_usns:
            target_scorecard_usn = direct_usns[0]
        elif is_scorecard_query or has_followup_reference:
            clean_name = re.sub(r"\b(?:gie|give|show|what|is|the|me|scorecard|score|card|profile|details|transcript|record|of|for|student|candidate|please|get)\b", " ", q_lower).strip()
            if clean_name and len(clean_name) >= 3:
                matched_cand = analytics.find_student(clean_name, recent_usns=recent_usns)
                if matched_cand:
                    target_scorecard_usn = matched_cand["usn"]
            if not target_scorecard_usn and recent_usns and (is_scorecard_query or has_followup_reference):
                target_scorecard_usn = recent_usns[0]

        if target_scorecard_usn:
            st = analytics.find_student(target_scorecard_usn)
            if st:
                sc = analytics.get_student_scorecard(target_scorecard_usn)
                if "error" not in sc:
                    additional_data += f"\n[OFFICIAL ACADEMIC SCORECARD FOR {sc['name']} ({sc['usn']})]:\n"
                    additional_data += f"Official Name: {sc['name']} | University Seat Number (USN): {sc['usn']}\n"
                    additional_data += f"Cohort Class Rank: #{sc['rank']} of 200 Candidates | Academic Standing: {sc['standing']}\n"
                    additional_data += f"Final Cumulative CGPA: {sc['cgpa']:.2f} / 10.0 | Active Backlogs: {sc['backlogs']}\n"
                    additional_data += f"Total Credits Earned: {sc['total_credits']} / 136.5 | 8-Semester Marks Average: {sc['average_marks']}/100\n"
                    additional_data += f"Semester SGPA Progression: {json.dumps(sc['sgpa_by_semester'])}\n"
                    additional_data += f"Best Semester: {sc['best_semester']} | Lowest: {sc['lowest_semester']}\n"
                    additional_data += f"Highest Scoring Course: {sc['highest_subject']}\n"
                    additional_data += f"Lowest Scoring Course: {sc['lowest_subject']}\n\n"
                    additional_data += f"COMPLETE 8-SEMESTER OFFICIAL COURSE-BY-COURSE MARKS (FROM DATABASE):\n"
                    for sm_idx in range(1, 9):
                        sm_data = st["semesters"][str(sm_idx)]
                        sub_strs = [f"{s['name']}: {s['marks']}/100 ({s['grade']})" for s in sm_data["subjects"]]
                        additional_data += f"• Semester {sm_idx} (SGPA: {sm_data['sgpa']:.2f}): {', '.join(sub_strs)}\n"
                    citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
                    citations_list.append("[Source: college_exam_results.pdf]")

    # 8. Check for Subject / Domain Academic Analytics (e.g. "Who is best at maths overall", "ML toppers", "OS average")
    subj_data = None
    if not additional_data:
        subj_data = analytics.get_subject_or_domain_analytics(q_lower)
        if subj_data:
            additional_data += f"\n{subj_data}\n"
            citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
            citations_list.append("[Source: college_exam_results.pdf]")

    # 9. Name Search (Detecting if multiple students share that name)
    if not additional_data and not any(w in q_lower for w in ["top", "average", "class", "policy", "grade", "menu", "fee", "compare", "fail", "arrear", "topper", "best", "highest", "lowest", "placement", "graduation"]):
        clean_name_q = re.sub(r"\b(?:who|is|tell|me|about|show|profile|details|student|candidate|please)\b", " ", q_lower).strip()
        name_matches = analytics.find_students_by_name(clean_name_q if len(clean_name_q) >= 3 else user_query)
        if len(name_matches) > 1:
            additional_data += f"\n[MULTIPLE CANDIDATES SHARING NAME '{user_query}' ({len(name_matches)} Students)]:\n"
            for st in name_matches:
                additional_data += f"- {st['name']} (USN: {st['usn']}) | Rank #{st['rank']} of 200 | CGPA: {st['cgpa']:.2f} | Standing: {st['standing']} | Backlogs: {st['backlogs']}\n"
            citations_list.append("[Source: college_exam_results.pdf, Page 1]")
            citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        elif len(name_matches) == 1:
            sc = analytics.get_student_scorecard(name_matches[0]["usn"])
            st = analytics.find_student(name_matches[0]["usn"])
            additional_data += f"\n[OFFICIAL SCORECARD FOR {sc['name']} ({sc['usn']})]:\n"
            additional_data += f"Official Name: {sc['name']} | USN: {sc['usn']}\n"
            additional_data += f"Rank: #{sc['rank']} of 200 Candidates | Standing: {sc['standing']}\n"
            additional_data += f"Cumulative CGPA: {sc['cgpa']:.2f} / 10.0 | Active Backlogs: {sc['backlogs']}\n"
            additional_data += f"Total Credits: {sc['total_credits']} | 8-Semester Marks Average: {sc['average_marks']}/100\n"
            additional_data += f"SGPA by Semester: {json.dumps(sc['sgpa_by_semester'])}\n"
            additional_data += f"Best Semester: {sc['best_semester']} | Lowest: {sc['lowest_semester']}\n"
            additional_data += f"Highest Subject: {sc['highest_subject']} | Lowest: {sc['lowest_subject']}\n"
            citations_list.append("[Source: students_results.json / Autonomous Examination Database]")

    # 10. Check for Overall Batch Top Scorers / Rank List (Support variable N e.g. top 3, top 5, top 10, toppers)
    is_top_query = bool(
        re.search(r"top\s*(?:\d+)?\s*(?:scorer|scorers|student|students|candidate|candidates|rank|topper|toppers|batch|cgpa)", q_lower)
        or any(keyword in q_lower for keyword in ["topper", "toppers", "rank 1", "highest cgpa", "merit list", "first rank", "highest score"])
    )
    if is_top_query and not subj_data:
        target_sem = extract_semester(q_lower)
        num_match = re.search(r"(?:top|first)\s*(\d+)", q_lower)
        n_count = int(num_match.group(1)) if num_match else 5
        
        top_data = analytics.get_top_scorers(n=n_count, semester=target_sem)
        if target_sem:
            additional_data += f"\n[OFFICIAL TOP {n_count} SCORERS - SEMESTER {target_sem} (RANKED STRICTLY BY SEMESTER SGPA)]:\n"
            for st in top_data["top_students"]:
                additional_data += f"Rank #{st['rank']}: {st['name']} (USN: {st['usn']}) - Semester {target_sem} SGPA: {st['sgpa']} (Cumulative CGPA: {st['overall_cgpa']})\n"
        else:
            additional_data += f"\n[OFFICIAL MERIT ROLL - TOP {n_count} BATCH SCORERS BY CGPA]:\n"
            for st in top_data["top_students"]:
                additional_data += f"Rank #{st['rank']}: {st['name']} (USN: {st['usn']}) - CGPA: {st['cgpa']} ({st['standing']})\n"
        citations_list.append("[Source: college_exam_results.pdf, Page 1]")
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")

    # 11. Check for Batch / Class Average & Statistics (if not asking for subject average)
    if any(keyword in q_lower for keyword in ["average", "class average", "batch average", "pass percentage", "overall stats", "statistics"]) and not subj_data:
        target_sem = extract_semester(q_lower)
        batch_stats = analytics.calculate_batch_analytics(semester=target_sem)
        if target_sem:
            additional_data += f"\n[EXACT SEMESTER {target_sem} BATCH ANALYTICS]:\n"
            additional_data += f"Average SGPA: {batch_stats['average_sgpa']} | Highest: {batch_stats['highest_sgpa']} | Lowest: {batch_stats['lowest_sgpa']}\n"
            additional_data += f"Pass Percentage: {batch_stats['pass_percentage']}% ({batch_stats['students_passed']} passed, {batch_stats['students_with_backlogs']} backlogs)\n"
        else:
            additional_data += "\n[EXACT 200-STUDENT BATCH OVERALL ANALYTICS]:\n"
            additional_data += f"Total Candidates: {batch_stats['total_candidates']} CSE Students\n"
            additional_data += f"Overall Batch Average CGPA: {batch_stats['average_cgpa']} / 10.0\n"
            additional_data += f"Highest CGPA: {batch_stats['highest_cgpa']} | Lowest CGPA: {batch_stats['lowest_cgpa']}\n"
            additional_data += f"Distinction Candidates (CGPA >= 8.5): {batch_stats['distinction_count']} ({batch_stats['distinction_pct']}%)\n"
            additional_data += f"First Class Candidates (6.75 - 8.49): {batch_stats['first_class_count']} ({batch_stats['first_class_pct']}%)\n"
            additional_data += f"Candidates with Backlogs: {batch_stats['candidates_with_backlogs']}\n"
            additional_data += f"Semester SGPA Averages: {json.dumps(batch_stats['semester_sgpa_averages'])}\n"
        citations_list.append("[Source: college_exam_results.pdf, Page 1]")
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")

    # 12. Check for Failed Students / Backlogs Query
    is_fail_query = any(keyword in q_lower for keyword in ["fail", "failed", "backlog", "backlogs", "arrear", "arrears"])
    if is_fail_query:
        target_sem = extract_semester(q_lower)
        fail_data = analytics.get_failed_students(semester=target_sem)
        if target_sem:
            additional_data += f"\n[OFFICIAL CANDIDATES WHO FAILED IN SEMESTER {target_sem} ({fail_data['count']} Students)]:\n"
            for st in fail_data["students"]:
                additional_data += f"- {st['name']} (USN: {st['usn']}) | Semester {target_sem} SGPA: {st['sgpa']:.2f} | Failed Course(s): {', '.join(st['failed_subjects'])}\n"
        else:
            additional_data += f"\n[OFFICIAL CANDIDATES WITH ACTIVE BACKLOGS ({fail_data['count']} Students)]:\n"
            for st in fail_data["students"]:
                additional_data += f"- {st['name']} (USN: {st['usn']}) | Cumulative CGPA: {st['cgpa']:.2f} | Active Backlogs: {st['backlogs_count']} | Standing: {st['standing']}\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf]")

    # 13. Check if user explicitly requested an Academic Visual Chart / Graph
    if visualizer.is_graph_requested(user_query):
        chart_res = visualizer.dispatch_graph_for_query(user_query, history=history, show=show_graph)
        if chart_res:
            additional_data += (
                f"\n[OFFICIAL VISUAL GRAPH GENERATED & DISPLAYED]:\n"
                f"Description: {chart_res['description']}\n"
                f"Interactive GUI Window: Rendered on screen (via matplotlib TkAgg backend)\n"
                f"Saved Artifact File: {chart_res['chart_path']}\n"
            )
            citations_list.append(f"[Visual Chart: {os.path.basename(chart_res['chart_path'])}]")

    # 7. Semantic Vector Retrieval from PDF via Chroma
    # Only retrieve from Chroma if additional_data does not already contain verified calculations/records,
    # avoiding random irrelevant PDF pages from polluting the grounded context.
    pdf_context = ""
    if not additional_data and not is_policy_query and not is_fail_query:
        pdf_context, citations = retrieve_with_sources(user_query, k=2)
        citations_list.extend(citations)

    # Combine calculation data + semantic PDF passages
    full_context = ""
    if additional_data:
        full_context += f"--- OFFICIAL AUTONOMOUS DATABASE CALCULATIONS (EXACT & VERIFIED ACROSS ALL 200 STUDENTS) ---\n{additional_data}\n\n"
        
    if pdf_context:
        full_context += f"--- RETRIEVED PDF TRANSCRIPT EXCERPTS ---\n{pdf_context}"
    
    unique_citations = list(dict.fromkeys(citations_list))
    citations_str = "\n".join(unique_citations) if unique_citations else "[Source: college_exam_results.pdf]"

    return full_context, citations_str

def print_erp_capabilities_menu():
    print("\n" + "=" * 80)
    print("  APEX INSTITUTE OF TECHNOLOGY - ACADEMIC ERP AI ASSISTANT DIRECTORY")
    print("=" * 80)
    print("""
[1] STUDENT SCORECARDS & TRANSCRIPTS:
  • Full 8-Semester Record : 'Show scorecard for student 1RV21CS028'
  • Semester Subject Marks : 'Show marks of 1RV21CS028 in Semester 5'
  • Name Search (with alerts): 'Who is Divya Bhat?' (handles duplicate names)

[2] HEAD-TO-HEAD CANDIDATE COMPARISONS:
  • Multi-Semester Compare : 'Compare student 1RV21CS028 and 1RV21CS007'
  • Conversational Followup: 'What are their respective names?' (k>=6 memory)
  • Metric Progression     : 'Who scored higher in Semester 5?' / 'Who has fewer backlogs?'

[3] SUBJECT MASTERY & DOMAIN ANALYTICS:
  • Multi-Course Domains   : 'Who is best at maths overall?' (Math I, Math II, Discrete Math)
  • Subject Toppers        : 'Who topped Machine Learning?' / 'Top scorer in Java Lab'
  • Class Averages & Pass  : 'What is the average marks in Operating Systems?'
  • Grade Distribution     : 'What is the grade distribution in Compiler Design?'

[4] MERIT ROLLS & TOPPERS:
  • Batch CGPA Honor Roll  : 'Who are the top 5 scorers in the batch?' / 'Top 10 toppers'
  • Semester SGPA Toppers  : 'Who topped Semester 4?' / 'Top 3 students in Sem 7'

[5] BATCH STATISTICS & TRENDS:
  • Cohort Average & Pass  : 'What is the overall batch average CGPA and pass percentage?'
  • Performance Brackets   : 'How many students achieved First Class with Distinction?'
  • Semester Progression   : 'What was the average SGPA in Semester 3?'

[6] TRAINING & PLACEMENT (TPO) SCREENING:
  • Tier-1 Shortlisting    : 'Who is eligible for Tier-1 placements (CGPA >= 8.5, 0 backlogs)?'
  • General Cutoff Drives  : 'List students eligible for placement with cutoff 7.5 CGPA'

[7] DEGREE AUDIT & GRADUATION CLEARANCE:
  • Individual Clearance   : 'Is student 1RV21CS028 eligible for graduation?'
  • Cohort Convocation     : 'How many students are cleared for degree conferment?'

[8] BACKLOG REGISTRY & REMEDIATION:
  • Cohort Active Arrears  : 'List all candidates with active backlogs'
  • Semester Failures      : 'List students who failed in Semester 1'

[9] ACADEMIC REGULATIONS & CITATIONS:
  • Grading Criteria       : 'What is the grading policy and criteria for Grade O and A+?'
  • Out-of-Context Defense : 'What is the cafeteria lunch menu?' (Polite refusal)
  • Audit Citations        : Every answer cites exact PDF Page numbers and database records.

[10] VISUAL CHARTS & GRAPHS (PANDAS, SEABORN & MATPLOTLIB):
  • Student Comparison     : 'Compare student 1RV21CS028 and 1RV21CS007 and give bar graph'
  • Semester Marks Plot    : 'Show bar graph of marks for 1RV21CS028 in Semester 5'
  • Follow-Up Graphing     : 'Give graph for previous conversation' (Visualizes prior topic)
  • Top Scorers Chart      : 'Plot bar chart of top 5 students in Semester 3'
  • Grade Distribution     : 'Give chart for Machine Learning grade distribution'
  • Cohort Progression     : 'Show bar graph of cohort average across all 8 semesters'
  • Placement Analytics    : 'Show placement eligibility cutoff graph'
""")
    print("=" * 80 + "\n")

def print_banner():
    print("=" * 80)
    print("  APEX INSTITUTE OF TECHNOLOGY - COLLEGE EXAMINATION ERP AI ASSISTANT")
    print("  B.Tech Computer Science & Engineering (Batch 2021-2025 | 200 Students)")
    print("  Powered by Local RAG (Chroma + Ollama Llama 3.2:3b)")
    print(f"  Conversational Memory: Active (Remembers last {MAX_HISTORY_TURNS} conversation turns)")
    print("  Interactive Visualizer: Enabled (Pandas + Seaborn + Matplotlib TkAgg)")
    print("=" * 80)
    print("Quick Navigation:")
    print("  • Type 'help' or 'menu' to see the complete list of all ERP operations.")
    print("  • Type 'q' or 'quit' to exit.")
    print("Try Asking:")
    print("  1. 'Compare student 1RV21CS028 and 1RV21CS007'")
    print("  2. 'Give graph for previous conversation' (Interactive visual popup)")
    print("  3. 'Show marks of 1RV21CS028 in Semester 5 with bar graph'")
    print("  4. 'Who is best at maths overall?'")
    print("  5. 'Who is eligible for Tier-1 company placements?'")
    print("  6. 'What is the grade distribution in Machine Learning?'")
    print("  7. 'Is student 1RV21CS028 eligible for graduation?'")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    print_banner()

    conversation_history = []

    while True:
        try:
            user_input = input("Ask a question about the college exam results (q to quit, help for menu): ").strip()
        except KeyboardInterrupt:
            print("\n(Session active. Type 'q' to quit, 'help' for menu)")
            continue
        except EOFError:
            print("\nSession ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Session ended.")
            break
        if user_input.lower() in ["help", "menu", "commands", "features", "options"]:
            print_erp_capabilities_menu()
            continue

        # 1. Check if user is answering a previous clarifying counter-question
        resolved_clarification = analytics.resolve_clarification_from_history(user_input, history=conversation_history)
        effective_query = resolved_clarification if resolved_clarification else user_input
        if resolved_clarification:
            print(f"\n[Interpreting clarification: '{user_input}' -> '{resolved_clarification}']")

        # 2. Check if the query is underspecified or ambiguous (and not already clarified)
        ambiguity = analytics.detect_query_ambiguity(effective_query, history=conversation_history) if not resolved_clarification else None
        if ambiguity:
            clarification_msg = analytics.format_clarification_question(ambiguity)
            print("\n" + "-" * 78)
            print(clarification_msg)
            print("-" * 78 + "\n")
            
            # Save clarification turn to conversation history so user's next response can resolve it
            conversation_history.append({
                "question": user_input,
                "answer": clarification_msg
            })
            continue

        if visualizer.is_graph_requested(effective_query):
            print("\nSearching PDF transcripts and computing academic visual analytics...")
        else:
            print("\nSearching PDF transcripts and computing analytics...")

        context_str, citations_str = route_and_build_context(effective_query, history=conversation_history, show_graph=False)
        history_str = format_chat_history(conversation_history, k=MAX_HISTORY_TURNS)

        print("\n" + "-" * 78)
        full_response = ""
        # Real-time token streaming for instantaneous responsive terminal experience
        for chunk in chain.stream({
            "chat_history": history_str,
            "context": context_str,
            "citations": citations_str,
            "question": effective_query
        }):
            full_response += chunk
            print(chunk, end="", flush=True)
        print("\n" + "-" * 78)

        if visualizer.is_graph_requested(effective_query):
            # Display graph window AFTER the information text has streamed completely
            visualizer.display_latest_chart()
            print("📊 Interactive graph displayed on screen and saved to charts/ directory.\n")
        else:
            print("")

        # Save to conversational memory buffer (maintains at least k=6 turns)
        conversation_history.append({
            "question": user_input,
            "answer": full_response.strip()
        })
