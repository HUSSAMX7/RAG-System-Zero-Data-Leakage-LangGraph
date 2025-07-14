
from langchain_openai import OpenAIEmbeddings


class Embedding :

    def __init__(self):
        
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    


    def get_embedding(self):

        return self.embedding
    