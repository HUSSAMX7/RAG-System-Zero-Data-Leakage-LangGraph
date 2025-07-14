from langchain.tools import tool


def build_retriever_tool(retriever):

    @tool
    def retriever_tool(query: str) -> str:    
        """
        This tool searches the vector database and returns relevant information.
        """
        
        docs = retriever.invoke(query)

        if not docs: 
            return "No relevant information found."
        
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    return retriever_tool


