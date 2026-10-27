import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
from main import route_and_build_context
import visualizer

print("=== TEST 1: Non-graph query ===")
ctx1, cites1 = route_and_build_context("what is the score of 1RV21CS028 in sem 5", history=[], show_graph=False)
assert "[OFFICIAL VISUAL GRAPH" not in ctx1, "Non-graph query should NOT trigger graph!"
print("Test 1 Passed: No graph generated for normal query.")

print("\n=== TEST 2: Direct Graph Query ===")
ctx2, cites2 = route_and_build_context("Show marks of 1RV21CS028 in Semester 5 with bar graph", history=[], show_graph=False)
assert "[OFFICIAL VISUAL GRAPH GENERATED & DISPLAYED]" in ctx2, "Graph query should include graph audit!"
assert any("sem_5_marks_1RV21CS028.png" in c for c in cites2.split("\n")), "Citation should cite chart file!"
print("Test 2 Passed: Direct semester marks graph generated and cited.")

print("\n=== TEST 3: Multi-turn Follow-up Graph for Previous Conversation ===")
history = [
    {
        "question": "Compare student 1RV21CS028 and 1RV21CS007",
        "answer": "Candidate 1: Divya Bhat (1RV21CS028) CGPA 9.72 vs Candidate 2: Divya Bhat (1RV21CS007) CGPA 9.62..."
    }
]
ctx3, cites3 = route_and_build_context("give groph for previous conversation", history=history, show_graph=False)
assert "[OFFICIAL VISUAL GRAPH GENERATED & DISPLAYED]" in ctx3, "Followup graph query should include graph audit!"
assert any("comparison_1RV21CS028_vs_1RV21CS007.png" in c for c in cites3.split("\n")), "Citation should cite comparison chart file!"
assert "[OFFICIAL SIDE-BY-SIDE CANDIDATE COMPARISON" in ctx3, "Context should contain comparison data for the students!"
print("Test 3 Passed: Multi-turn followup 'give groph for previous conversation' successfully plotted comparison and retrieved comparison data.")

print("\nALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
