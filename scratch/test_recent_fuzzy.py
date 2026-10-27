import sys, difflib
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
import analytics

students = analytics._load_data()
st_map = {s["usn"]: s for s in students}

recent_usns = ["1RV21CS021", "1RV21CS008"]

query_name = "aditya verma"
query_tokens = query_name.lower().split()

print("Recent students:", [st_map[u]["name"] + " (" + u + ")" for u in recent_usns])

best_recent = None
best_ratio = 0.0

for u in recent_usns:
    st = st_map[u]
    st_name = st["name"].lower()
    ratio = difflib.SequenceMatcher(None, query_name, st_name).ratio()
    st_parts = st_name.split()
    token_overlap = any(tok in st_parts for tok in query_tokens)
    print(f"Comparison with {st['name']}: ratio={ratio:.2f}, token_overlap={token_overlap}")
    if (token_overlap or ratio > 0.6) and ratio > best_ratio:
        best_ratio = ratio
        best_recent = st

print("Best match from recent history:", best_recent["name"] if best_recent else None)
