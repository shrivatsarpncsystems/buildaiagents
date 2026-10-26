from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2:3b")

template = """You are reviewer of pizza restaurant 
here are some releant reviews: {reviews}
now based on these reviews, here are questions to answer:{questions} 
""" 

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input ("Ask your question (q to quit): ")
    if question.lower() == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews,"questions": question})
    print(result)
    print("\n")

    

