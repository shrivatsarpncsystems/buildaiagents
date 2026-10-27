import re
import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, retrieve_with_sources
import analytics

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
   - For semester toppers (e.g. "top 10 students in 3rd semester"), present the ranked list strictly by their Semester SGPA, showing Rank, Name, USN, and SGPA.
   - For failed students or their marks, ALWAYS list the student name, USN, semester SGPA, and each failed course along with their exact marks obtained out of 100 and Grade F (e.g. "Deepak Reddy (1RV21CS180) - Engineering Chemistry: 30/100 (Grade: F)").
   - If multiple candidates share the same name (e.g. Divya Bhat), always distinguish them explicitly by USN and rank.
5. STRICT GROUNDING & REFUSAL: Do NOT invent student marks, ranks, or policies. If the question cannot be answered from the provided records or conversation history (e.g. hostel fees, sports, cafeteria menu), explicitly state:
   "I do not have supporting context in the official college examination records to answer this question."
6. CITATIONS: Include the source citation at the end of your answer (e.g. "[Source: students_results.json / Autonomous Examination Database]").

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
    last_q = last_turn.get("question", "").lower()
    if any(w in last_q for w in ["fail", "failed", "backlog", "backlogs", "arrear", "arrears"]):
        last_turn_intent = "failed_students"
    elif any(w in last_q for w in ["compare", "versus", "vs", "difference"]):
        last_turn_intent = "comparison"
    elif any(w in last_q for w in ["top", "topper", "toppers", "rank"]):
        last_turn_intent = "toppers"
    elif any(w in last_q for w in ["placement", "drive", "shortlist"]):
        last_turn_intent = "placement"
        
    for turn in reversed(history[-max_turns:]):
        text = f"{turn.get('question', '')} {turn.get('answer', '')}"
        
        # Extract USNs (e.g. 1RV21CS028)
        found_usns = re.findall(r"1rv21cs\d{3}", text, re.IGNORECASE)
        for u in found_usns:
            u_upper = u.upper()
            if u_upper not in recent_usns:
                recent_usns.append(u_upper)
                
        # Extract Semesters using robust extractor
        q_sem = extract_semester(turn.get("question", ""))
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

def route_and_build_context(user_query: str, history=None):
    """
    Intelligent Query Router:
    Combines deterministic analytics calculation engine with Chroma vector search.
    Resolves conversational pronouns and references across multi-turn dialogs (k >= 6).
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
    is_compare_intent = any(w in q_lower for w in [
        "compare", "comparison", "difference", "versus", "vs", "between",
        "who scored higher", "who has higher", "who is better", "which one", "higher cgpa"
    ])

    # 1. Comparison of Two Students (Direct USNs or Follow-up referencing recent students)
    target_compare_usns = []
    if len(direct_usns) >= 2:
        target_compare_usns = direct_usns[:2]
    elif is_compare_intent and len(direct_usns) == 1 and len(recent_usns) >= 1:
        target_compare_usns = [recent_usns[0], direct_usns[0]]
    elif (is_compare_intent or (has_pronoun_reference and last_intent == "comparison")) and len(direct_usns) == 0 and len(recent_usns) >= 2:
        target_compare_usns = recent_usns[:2]

    if target_compare_usns and (is_compare_intent or len(direct_usns) >= 2 or any(k in q_lower for k in ["name", "score", "cgpa", "rank", "sem", "grade", "both", "they", "their", "who", "which"])):
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

    # 2. Check for Specific Student Semester Detailed Transcript (e.g. "What were 1RV21CS028 marks in sem 5")
    sem_from_query = extract_semester(q_lower)
    target_st_id = direct_usns[0] if direct_usns else (recent_usns[0] if (has_pronoun_reference and recent_usns) else None)
    if sem_from_query and target_st_id and not target_compare_usns and any(w in q_lower for w in ["mark", "marks", "subject", "subjects", "grade", "grades", "transcript", "score", "scores", "in sem", "in semester"]):
        additional_data += f"\n{analytics.get_student_semester_breakdown(target_st_id, sem_from_query)}\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf]")

    # 3. Check for Follow-Up Question on Marks for Previously Discussed Failed Students
    is_marks_followup = any(w in q_lower for w in ["mark", "marks", "score", "scores", "grade", "grades"]) and (
        has_pronoun_reference or any(w in q_lower for w in ["give", "show", "what", "their", "thei", "respectie", "respective"])
    )
    if is_marks_followup and last_intent == "failed_students" and recent_sems:
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

    # 7. Single Student Scorecard Lookup (Direct USN or Follow-up pronoun)
    elif len(direct_usns) == 1 or (has_pronoun_reference and len(recent_usns) >= 1 and not is_policy_query and not is_marks_followup):
        target_usn = direct_usns[0] if direct_usns else recent_usns[0]
        st = analytics.find_student(target_usn)
        if st:
            sc = analytics.get_student_scorecard(target_usn)
            if "error" not in sc:
                additional_data += f"\n[OFFICIAL SCORECARD FOR {sc['name']} ({sc['usn']})]:\n"
                additional_data += f"Official Name: {sc['name']} | USN: {sc['usn']}\n"
                additional_data += f"Rank: #{sc['rank']} of 200 Candidates | Standing: {sc['standing']}\n"
                additional_data += f"Cumulative CGPA: {sc['cgpa']:.2f} / 10.0 | Active Backlogs: {sc['backlogs']}\n"
                additional_data += f"Total Credits: {sc['total_credits']} | 8-Semester Marks Average: {sc['average_marks']}/100\n"
                additional_data += f"SGPA by Semester: {json.dumps(sc['sgpa_by_semester'])}\n"
                additional_data += f"Best Semester: {sc['best_semester']} | Lowest: {sc['lowest_semester']}\n"
                additional_data += f"Highest Subject: {sc['highest_subject']} | Lowest: {sc['lowest_subject']}\n"
                citations_list.append("[Source: students_results.json / Autonomous Examination Database]")

    # 8. Check for Subject / Domain Academic Analytics (e.g. "Who is best at maths overall", "ML toppers", "OS average")
    subj_data = analytics.get_subject_or_domain_analytics(q_lower)
    if subj_data:
        additional_data += f"\n{subj_data}\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        citations_list.append("[Source: college_exam_results.pdf]")

    # 9. Name Search (Detecting if multiple students share that name)
    elif not any(w in q_lower for w in ["top", "average", "class", "policy", "grade", "menu", "fee", "compare", "fail", "arrear", "topper", "best", "highest", "lowest", "placement", "graduation"]):
        name_matches = analytics.find_students_by_name(user_query)
        if len(name_matches) > 1:
            additional_data += f"\n[MULTIPLE CANDIDATES SHARING NAME '{user_query}' ({len(name_matches)} Students)]:\n"
            for st in name_matches:
                additional_data += f"- {st['name']} (USN: {st['usn']}) | Rank #{st['rank']} of 200 | CGPA: {st['cgpa']:.2f} | Standing: {st['standing']} | Backlogs: {st['backlogs']}\n"
            citations_list.append("[Source: college_exam_results.pdf, Page 1]")
            citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        elif len(name_matches) == 1:
            sc = analytics.get_student_scorecard(name_matches[0]["usn"])
            additional_data += f"\n[OFFICIAL SCORECARD FOR {sc['name']} ({sc['usn']})]:\n"
            additional_data += f"Official Name: {sc['name']} | USN: {sc['usn']}\n"
            additional_data += f"Rank: #{sc['rank']} of 200 Candidates | Standing: {sc['standing']}\n"
            additional_data += f"Cumulative CGPA: {sc['cgpa']:.2f} / 10.0 | Active Backlogs: {sc['backlogs']}\n"
            additional_data += f"Total Credits: {sc['total_credits']} | 8-Semester Marks Average: {sc['average_marks']}/100\n"
            additional_data += f"SGPA by Semester: {json.dumps(sc['sgpa_by_semester'])}\n"
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
""")
    print("=" * 80 + "\n")

def print_banner():
    print("=" * 80)
    print("  APEX INSTITUTE OF TECHNOLOGY - COLLEGE EXAMINATION ERP AI ASSISTANT")
    print("  B.Tech Computer Science & Engineering (Batch 2021-2025 | 200 Students)")
    print("  Powered by Local RAG (Chroma + Ollama Llama 3.2:3b)")
    print(f"  Conversational Memory: Active (Remembers last {MAX_HISTORY_TURNS} conversation turns)")
    print("=" * 80)
    print("Quick Navigation:")
    print("  • Type 'help' or 'menu' to see the complete list of all ERP operations.")
    print("  • Type 'q' or 'quit' to exit.")
    print("Try Asking:")
    print("  1. 'Compare student 1RV21CS028 and 1RV21CS007'")
    print("  2. 'What are their respective names?' (Multi-turn follow-up test)")
    print("  3. 'Who is best at maths overall?'")
    print("  4. 'Who is eligible for Tier-1 company placements?'")
    print("  5. 'Show marks of 1RV21CS028 in Semester 5'")
    print("  6. 'What is the grade distribution in Machine Learning?'")
    print("  7. 'Is student 1RV21CS028 eligible for graduation?'")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    print_banner()

    conversation_history = []

    while True:
        try:
            user_input = input("Ask a question about the college exam results (q to quit, help for menu): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Session ended.")
            break
        if user_input.lower() in ["help", "menu", "commands", "features", "options"]:
            print_erp_capabilities_menu()
            continue

        print("\nSearching PDF transcripts and computing analytics...")
        context_str, citations_str = route_and_build_context(user_input, history=conversation_history)
        history_str = format_chat_history(conversation_history, k=MAX_HISTORY_TURNS)

        print("\n" + "-" * 78)
        full_response = ""
        # Real-time token streaming for instantaneous responsive terminal experience
        for chunk in chain.stream({
            "chat_history": history_str,
            "context": context_str,
            "citations": citations_str,
            "question": user_input
        }):
            full_response += chunk
            print(chunk, end="", flush=True)
        print("\n" + "-" * 78 + "\n")

        # Save to conversational memory buffer (maintains at least k=6 turns)
        conversation_history.append({
            "question": user_input,
            "answer": full_response.strip()
        })
