
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():

    #return OpenAIEmbeddings(
     #   model="text-embedding-3-small"
    #)
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )