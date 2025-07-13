from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List



class Load :

    @staticmethod
    def load_files(pdf_path : str) -> List[Document]:

        loader = PyPDFLoader(pdf_path)
        
        return loader.load()
