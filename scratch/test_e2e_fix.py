import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
import main
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b", temperature=0.0)

history = []

print("=== TEST 1: Comparison ===")
q1 = "Compare student 1RV21CS021 and 1RV21CS008"
prompt1, cites1 = main.route_and_build_context(q1, history=history, show_graph=False)
res1 = llm.invoke(prompt1)
print(res1)
history.append({"query": q1, "response": res1})

print("\n=== TEST 2: Scorecard ===")
q2 = "gie me Aditya Verma score card"
prompt2, cites2 = main.route_and_build_context(q2, history=history, show_graph=False)
res2 = llm.invoke(prompt2)
print(res2)
