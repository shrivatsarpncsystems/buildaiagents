import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
import re, json
import analytics

def test_scorecard_routing(q_lower, history=[]):
    direct_usns = [u.upper() for u in list(dict.fromkeys(re.findall(r"1rv21cs\d{3}", q_lower)))]
    recent_usns = []
    if history:
        for turn in reversed(history[-6:]):
            for u in re.findall(r"1rv21cs\d{3}", f"{turn.get('question','')} {turn.get('answer','')}".lower()):
                if u.upper() not in recent_usns:
                    recent_usns.append(u.upper())

    is_scorecard_query = any(w in q_lower for w in ["scorecard", "score card", "profile", "record", "all semester", "all sem", "marks average", "performance"])
    target_scorecard_usn = None
    
    if direct_usns:
        target_scorecard_usn = direct_usns[0]
    elif is_scorecard_query or recent_usns:
        clean_name = re.sub(r"\b(?:gie|give|show|what|is|the|me|scorecard|score|card|profile|details|transcript|record|of|for|student|candidate|please|get)\b", " ", q_lower).strip()
        if clean_name and len(clean_name) >= 3:
            matches = analytics.find_students_by_name(clean_name)
            if matches:
                target_scorecard_usn = matches[0]["usn"]
        if not target_scorecard_usn and recent_usns:
            target_scorecard_usn = recent_usns[0]
            
    print("Resolved target_scorecard_usn:", target_scorecard_usn)
    if target_scorecard_usn:
        sc = analytics.get_student_scorecard(target_scorecard_usn)
        print("Scorecard name:", sc["name"])
        print("Scorecard CGPA:", sc["cgpa"])
        print("Scorecard Rank:", sc["rank"])
        print("Sample Sem 1 subjects from database:", [f"{s['name']}: {s['marks']}/100" for s in sc["semesters_detail"]["1"]["subjects"][:3]])

test_scorecard_routing("gie me Aditya Verma score card", history=[{"question": "Compare student 1RV21CS021 and 1RV21CS008", "answer": "..."}])
