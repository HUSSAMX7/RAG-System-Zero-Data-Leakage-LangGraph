from langchain_openai import ChatOpenAI

def get_llm():

    return ChatOpenAI(model="gpt-4o")


