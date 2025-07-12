
from langchain_openai import ChatOpenAI


class llm:

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature = 0)

    
    def get_llm(self):
        
        return self.model
