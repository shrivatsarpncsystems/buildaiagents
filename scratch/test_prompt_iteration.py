import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from main import route_and_build_context

test_template = """You are the Official Academic AI Assistant for Apex Institute of Technology (Department of Computer Science & Engineering).

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
1. ABSOLUTELY NO META-TALK: NEVER start responses with "Based on the provided conversation history", "Based on the academic context", or conversational disclaimers. Start DIRECTLY with the answer.
2. OFFICIAL CALCULATIONS PRECEDENCE: The ACADEMIC CONTEXT contains exact, verified calculations across all 200 students. You MUST use those exact details, marks, ranks, and numbers.
3. SEMESTER MARKS & PERCENTAGE FORMATTING (MANDATORY):
   When asked for marks, score, transcript, or percentage of a student in a semester, you MUST output ALL of the following:
   a. Candidate Header: Student Name, USN, and Semester SGPA.
   b. Individual Subject Marks: List EVERY subject individually with its course name, marks obtained out of 100, individual percentage (e.g. 93.0%), and grade.
   c. Total Marks & Percentage: State the Total Marks Obtained out of maximum marks (e.g. 562 / 600) and the Overall Semester Percentage (e.g. 93.67%).
4. CITATIONS: Include the source citation at the end of your answer (e.g. "[Source: students_results.json / Autonomous Examination Database]").

OFFICIAL RESPONSE:"""

model = OllamaLLM(model="llama3.2:3b", temperature=0.0)
prompt = ChatPromptTemplate.from_template(test_template)
chain = prompt | model

query = "can you calculate marks of 1RV21CS028 in sem 5 and give in percentage format"
context, citations = route_and_build_context(query)
cite_str = "\n".join(citations)

print("Invoking prompt...")
res = chain.invoke({
    "context": context,
    "citations": cite_str,
    "chat_history": "No previous conversation history.",
    "question": query
})

print("\n=== ITERATION RESULT ===")
print(res)
