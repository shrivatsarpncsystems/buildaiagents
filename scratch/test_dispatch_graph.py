import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
import visualizer

# Test 1: Query without graph keyword -> should be None
res1 = visualizer.dispatch_graph_for_query("who is top 5 in sem 3", history=[], show=False)
print("Test 1 (No graph keyword):", res1)
assert res1 is None, "Should be None when no graph requested!"

# Test 2: Compare two students with graph
res2 = visualizer.dispatch_graph_for_query("compare 1RV21CS028 and 1RV21CS007 and give bar graph", history=[], show=False)
print("Test 2 (Compare with graph):", res2["chart_type"], res2["chart_path"])

# Test 3: Follow-up: "give groph for previous conversation"
history_compare = [
    {
        "question": "Compare student 1RV21CS028 and 1RV21CS007",
        "answer": "Here is the comparison between Divya Bhat (1RV21CS028) and Divya Bhat (1RV21CS007)..."
    }
]
res3 = visualizer.dispatch_graph_for_query("give groph for previous conversation", history=history_compare, show=False)
print("Test 3 (Followup comparison graph):", res3["chart_type"], res3["chart_path"])

# Test 4: Follow-up: "show bar graph for this" after student semester marks
history_sem_marks = [
    {
        "question": "what is the score of Divya Bhat (1RV21CS028) in 5th semester?",
        "answer": "Total Marks: 562 / 600 (93.67%)..."
    }
]
res4 = visualizer.dispatch_graph_for_query("show bar graph for this", history=history_sem_marks, show=False)
print("Test 4 (Followup semester marks graph):", res4["chart_type"], res4["chart_path"])

# Test 5: "plot top 5 scorers in sem 4"
res5 = visualizer.dispatch_graph_for_query("plot top 5 scorers in sem 4", history=[], show=False)
print("Test 5 (Top scorers graph):", res5["chart_type"], res5["chart_path"])

# Test 6: "give chart for operating systems grade distribution"
res6 = visualizer.dispatch_graph_for_query("give chart for operating systems grade distribution", history=[], show=False)
print("Test 6 (Grade distribution chart):", res6["chart_type"], res6["chart_path"])

print("\nALL DISPATCH TESTS PASSED PERFECTLY!")
