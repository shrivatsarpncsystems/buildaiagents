import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
import visualizer

print("Testing plot_student_semester_marks...")
p1 = visualizer.plot_student_semester_marks("1RV21CS028", 5, show=False)
print("P1:", p1)

print("Testing plot_student_comparison...")
p2 = visualizer.plot_student_comparison("1RV21CS028", "1RV21CS007", show=False)
print("P2:", p2)

print("Testing plot_student_scorecard...")
p3 = visualizer.plot_student_scorecard("1RV21CS070", show=False)
print("P3:", p3)

print("Testing plot_top_scorers...")
p4 = visualizer.plot_top_scorers(5, semester=3, show=False)
print("P4:", p4)

print("Testing plot_grade_distribution...")
p5 = visualizer.plot_grade_distribution("machine learning", show=False)
print("P5:", p5)

print("Testing plot_failed_students...")
p6 = visualizer.plot_failed_students(semester=2, show=False)
print("P6:", p6)

print("Testing plot_batch_averages...")
p7 = visualizer.plot_batch_averages(show=False)
print("P7:", p7)

print("Testing plot_placement_eligibility...")
p8 = visualizer.plot_placement_eligibility(min_cgpa=8.0, show=False)
print("P8:", p8)

print("\nALL VISUALIZER FUNCTIONS EXECUTED SUCCESSFULLY!")
