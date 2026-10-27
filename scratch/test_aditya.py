import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
import analytics

print("Searching 'Aditya Verma':")
m1 = analytics.find_students_by_name("Aditya Verma")
print("Matches for 'Aditya Verma':", [s["name"] + " (" + s["usn"] + ")" for s in m1])

print("\nSearching 'gie me Aditya Verma score card':")
m2 = analytics.find_students_by_name("gie me Aditya Verma score card")
print("Matches for full query:", [s["name"] + " (" + s["usn"] + ")" for s in m2])

print("\nAll students with Verma:")
v = [s["name"] + " (" + s["usn"] + ")" for s in analytics._load_data() if "verma" in s["name"].lower()]
print("Verma students:", v)

print("\nAll students with Aditya:")
a = [s["name"] + " (" + s["usn"] + ")" for s in analytics._load_data() if "aditya" in s["name"].lower()]
print("Aditya students:", a)
