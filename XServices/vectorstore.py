import os
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List, Optional
import shutil



class VectorStoreService:
    def __init__(self, assistant_id: str, file_path: str, embed_model: Embeddings):
        self.assistant_id = assistant_id
        self.file_path = file_path
        self.embed_model = embed_model
        self.vectorstore_path = self._build_vectorstore_path()


    def _build_vectorstore_path(self) -> str:
        
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]
        path = os.path.join("VectorStorage",str(self.assistant_id),file_name)
        os.makedirs(path,exist_ok=True)

        return path
    
    
    
    def get_or_build(self, documents: Optional[List[Document]] = None) -> FAISS:
        """ 
        first here we load the vectorestore if we dont have we will create  
        """
        if self._vectorstore_exists():
            return FAISS.load_local(
                folder_path=self.vectorstore_path,
                embeddings=self.embed_model,
                allow_dangerous_deserialization=True  
            )
        
        if documents is None or len(documents) == 0:
            raise ValueError("No vectorstore found and no documents provided to build one.")
        
    
        vectorstore = FAISS.from_documents(documents, self.embed_model)
        vectorstore.save_local(folder_path=self.vectorstore_path)
        return vectorstore
    
    

    def _vectorstore_exists(self) -> bool:
        index_path = os.path.join(self.vectorstore_path, "index.faiss")
        return os.path.exists(index_path)
    
    @staticmethod
    def delete_by_file_name(assistant_id:str, file_name:str):
        
        """
        Delete a specific vectorstore by category and file name (without extension).
        Example: category="hr", file_name="ai_eng" -> VectorStorage/hr/ai_eng/
        """
        
        path = os.path.join("VectorStorage", str(assistant_id), file_name)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f" Deleted vectorstore at: {path}")
        else:
            print(f"No vectorstore found at: {path}")
