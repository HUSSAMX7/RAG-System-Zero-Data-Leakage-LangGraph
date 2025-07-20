from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List

class VectorStore:

    @staticmethod
    def build_vectorstore(
        documents: List[Document],
        embeddings: Embeddings,
    ) -> Chroma:
    
        try:

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=r"./local_vdb",
                collection_name="data" # 
                )
            
            return vectorstore.as_retriever(
                search_type= "similarity",
                search_kwargs={"k": 5}
            )
            
        except Exception as e:
            print(f"Erorr setting up {str(e)}")
            raise