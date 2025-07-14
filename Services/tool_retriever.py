from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever



def create_retriever_tool(retriever: VectorStoreRetriever):
    
    @tool
    def retriever_tool(query: str) -> str:
        """
        Search the loaded PDF and return relevant information based on the query.
        """
        docs = retriever.invoke(query)

        if not docs:
            return "I found no relevant information."

        return "\n\n".join(
            [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

    return retriever_tool
