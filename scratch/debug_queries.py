import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
from main import route_and_build_context, format_chat_history, chain
import analytics

print("==================== DEBUG 1: Compare student 1RV21CS021 and 1RV21CS008 ====================")
ctx1, cites1 = route_and_build_context("Compare student 1RV21CS021 and 1RV21CS008", history=[], show_graph=False)
print("--- CONTEXT 1 ---")
print(ctx1)

print("\n==================== DEBUG 2: gie me Aditya Verma score card ====================")
# Turn 1 history
h1 = [
    {
        "question": "Compare student 1RV21CS021 and 1RV21CS008",
        "answer": "Candidate 1 (Aditi Verma, 1RV21CS021) has a higher cumulative CGPA (7.95 / 10.0) compared to Candidate 2 (Harsh Mehta, 1RV21CS008) (2.96 / 10.0)."
    }
]
ctx2, cites2 = route_and_build_context("gie me Aditya Verma score card", history=h1, show_graph=False)
print("--- CONTEXT 2 ---")
print(ctx2)
