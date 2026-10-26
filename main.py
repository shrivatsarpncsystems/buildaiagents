import re
import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, retrieve_with_sources
import analytics

# 1. Initialize local LLM
model = OllamaLLM(model="llama3.2:3b", temperature=0.2)

# 2. Strict RAG Prompt Template with Grounding, Citations & Handling Uncontextualized Queries
template = """You are the Official Academic AI Assistant for Apex Institute of Technology (Department of Computer Science & Engineering).
Your role is to assist students, faculty, and administrators with examination records, transcripts, SGPA/CGPA calculations, merit lists, and academic regulations.

==================================================
ACADEMIC CONTEXT & DATA:
{context}

SOURCE CITATIONS AVAILABLE:
{citations}
==================================================

USER QUESTION: {question}

STRICT INSTRUCTIONS FOR YOUR RESPONSE:
1. FACTUAL GROUNDING: Rely strictly and accurately on the provided ACADEMIC CONTEXT and calculations. Do NOT invent student marks, ranks, or policies.
2. SOURCE CITATIONS: Always include clear source citations at the end of your answer referencing the specific document and page numbers from the citations provided (e.g., "[Source: college_exam_results.pdf, Page X]").
3. HANDLING QUESTIONS WITHOUT SUPPORTING CONTEXT: If the user's question cannot be answered from the provided records (e.g. asking about hostel fees, sports, cafeteria, or an unlisted student), explicitly state:
   "I do not have supporting context in the official college examination records to answer this question."
4. PRESENTATION: Present numbers, grades, SGPA/CGPA, and rankings clearly using clean bullet points, tables, or highlighted summaries.

OFFICIAL RESPONSE:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def route_and_build_context(user_query: str):
    """
    Intelligent Query Router:
    Combines deterministic analytics calculation engine with Chroma vector search.
    Provides exact mathematical precision for averages and rankings, 
    plus semantic search for transcripts, syllabus, and policies.
    """
    q_lower = user_query.lower().strip()
    additional_data = ""
    citations_list = []

    # 1. Check for Student USN (e.g. 1RV21CS042)
    usn_match = re.search(r"1rv21cs\d{3}", q_lower)
    if usn_match:
        usn = usn_match.group(0).upper()
        sc = analytics.get_student_scorecard(usn)
        if "error" not in sc:
            additional_data += f"\n[CALCULATED SCORECARD FOR {usn}]:\n"
            additional_data += f"Student Name: {sc['name']} | Rank: #{sc['rank']} of 200\n"
            additional_data += f"Cumulative CGPA: {sc['cgpa']:.2f} / 10.0 | Academic Standing: {sc['standing']}\n"
            additional_data += f"Total Credits Completed: {sc['total_credits']} | Active Backlogs: {sc['backlogs']}\n"
            additional_data += f"SGPA Progression: {json.dumps(sc['sgpa_by_semester'])}\n"
            additional_data += f"Best Semester: {sc['best_semester']} | Lowest Semester: {sc['lowest_semester']}\n"
            additional_data += f"Average Mark across 8 semesters: {sc['average_marks']}/100\n"
            additional_data += f"Highest Subject Score: {sc['highest_subject']}\n"

    # 2. Check for Top Scorers / Rank List
    if any(keyword in q_lower for keyword in ["top scorer", "topper", "toppers", "rank 1", "highest cgpa", "merit list", "first rank"]):
        # Check if asking for specific semester
        sem_match = re.search(r"(?:sem|semester)\s*([1-8])", q_lower)
        target_sem = int(sem_match.group(1)) if sem_match else None
        
        top_data = analytics.get_top_scorers(n=5, semester=target_sem)
        if target_sem:
            additional_data += f"\n[OFFICIAL TOP 5 SCORERS - SEMESTER {target_sem}]:\n"
            for st in top_data["top_students"]:
                additional_data += f"Rank {st['rank']}: {st['name']} ({st['usn']}) - SGPA: {st['sgpa']} (Overall CGPA: {st['overall_cgpa']})\n"
        else:
            additional_data += "\n[OFFICIAL MERIT ROLL - TOP 5 BATCH SCORERS BY CGPA]:\n"
            for st in top_data["top_students"]:
                additional_data += f"Rank {st['rank']}: {st['name']} ({st['usn']}) - CGPA: {st['cgpa']} ({st['standing']})\n"

    # 3. Check for Batch / Class Average & Statistics
    if any(keyword in q_lower for keyword in ["average", "class average", "batch average", "pass percentage", "overall stats", "statistics"]):
        sem_match = re.search(r"(?:sem|semester)\s*([1-8])", q_lower)
        target_sem = int(sem_match.group(1)) if sem_match else None
        
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

    # 4. Check for Comparison between Two Students (e.g. "compare 1RV21CS001 and 1RV21CS002")
    compare_matches = re.findall(r"1rv21cs\d{3}", q_lower)
    if len(compare_matches) >= 2:
        comp_res = analytics.compare_students(compare_matches[0].upper(), compare_matches[1].upper())
        if "error" not in comp_res:
            additional_data += f"\n[STUDENT COMPARISON: {comp_res['student1']['name']} vs {comp_res['student2']['name']}]:\n"
            additional_data += f"{comp_res['student1']['name']}: Rank #{comp_res['student1']['rank']}, CGPA {comp_res['student1']['cgpa']}\n"
            additional_data += f"{comp_res['student2']['name']}: Rank #{comp_res['student2']['rank']}, CGPA {comp_res['student2']['cgpa']}\n"
            additional_data += f"CGPA Difference: {comp_res['cgpa_difference']} (Higher: {comp_res['cgpa_winner']})\n"

    # 5. Semantic Vector Retrieval from PDF via Chroma (k=2 for fast responsive local LLM inference)
    pdf_context, citations = retrieve_with_sources(user_query, k=2)
    citations_list.extend(citations)

    # Combine calculation data + semantic PDF passages
    full_context = ""
    if additional_data:
        full_context += f"--- CALCULATED DATABASE METRICS ---\n{additional_data}\n\n"
        citations_list.append("[Source: students_results.json / Autonomous Examination Database]")
        
    full_context += f"--- RETRIEVED PDF TRANSCRIPT EXCERPTS ---\n{pdf_context}"
    
    unique_citations = list(dict.fromkeys(citations_list))
    citations_str = "\n".join(unique_citations) if unique_citations else "[Source: college_exam_results.pdf]"

    return full_context, citations_str

def print_banner():
    print("=" * 78)
    print("  APEX INSTITUTE OF TECHNOLOGY - COLLEGE EXAM RESULTS AI ASSISTANT")
    print("  B.Tech Computer Science & Engineering (Batch 2021-2025 | 200 Students)")
    print("  Powered by Local RAG (Chroma + Ollama Llama 3.2:3b)")
    print("=" * 78)
    print("Quick Action Suggestions:")
    print("  1. 'Who are the top 5 scorers in the batch?'")
    print("  2. 'Who topped Semester 4?'")
    print("  3. 'What is the overall batch average CGPA and pass percentage?'")
    print("  4. 'Show scorecard for student 1RV21CS028'")
    print("  5. 'Compare student 1RV21CS001 and 1RV21CS002'")
    print("  6. 'What is the grading policy and criteria for Grade O and A+?'")
    print("  7. 'What is the cafeteria lunch menu?' (Test out-of-context handling)")
    print("  Type 'q' to exit.")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    print_banner()

    while True:
        try:
            user_input = input("Ask a question about the college exam results (q to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Session ended.")
            break

        print("\nSearching PDF transcripts and computing analytics...")
        context_str, citations_str = route_and_build_context(user_input)

        response = chain.invoke({
            "context": context_str,
            "citations": citations_str,
            "question": user_input
        })

        print("\n" + "-" * 78)
        print(response.strip())
        print("-" * 78 + "\n")
