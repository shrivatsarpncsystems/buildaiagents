import sys
sys.path.insert(0, r"c:\Users\User\OneDrive - RPNC Systems Inc\Documents\buildaiagent")
from main import chain, route_and_build_context

query = "can you calculate marks of 1RV21CS028 in sem 5 and give in percentage format"
context, citations = route_and_build_context(query)
cite_str = "\n".join(citations)

print("Invoking LLM...")
res = chain.invoke({
    "context": context,
    "citations": cite_str,
    "chat_history": "No previous conversation history.",
    "question": query
})

print("\n=== FINAL LLM RESPONSE ===")
print(res)
