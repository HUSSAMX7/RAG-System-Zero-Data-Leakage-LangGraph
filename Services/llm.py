from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_llm():

    #return ChatOpenAI(model="gpt-4o")
        return ChatOllama(model="llama3:8b", temperature=0.1)


