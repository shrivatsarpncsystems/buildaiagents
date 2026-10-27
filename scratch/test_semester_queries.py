import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
from main import route_and_build_context

test_queries = [
    "can you calculate marks of 1RV21CS028 in sem 5 and give in percentage format",
    "what is the score of Divya Bhat in 5th semester?",
    "score of Yash Verma in sem 3",
    "marks of 1RV21CS070 in 4th semister",
]

for idx, q in enumerate(test_queries, 1):
    print(f"\n==================== TEST {idx}: {q} ====================")
    data, cites = route_and_build_context(q)
    print("--- CONTEXT DATA ---")
    print(data.strip())
    print("--- CITATIONS ---")
    print(cites)
