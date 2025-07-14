from services.embedding import Embedding
from services.llm import LLM
from services.memory import Memory
from services.load import Load
from services.text_splitter import TextSpliter
from services.vectorstore import VectorStoreService
from services.graph_builder import GraphBuilder
from prompts.prompt import Prompt
from langchain_core.messages import HumanMessage
from langchain.tools.retriever import create_retriever_tool

def main():
    file_path = "C:/Users/hosam/Desktop/ai.pdf"
    
    # تحميل وتقسيم
    documents = TextSpliter.split_documents(Load.load_files(file_path))
    
    # بناء متجر
    vectorstore = VectorStoreService(file_path).get_or_build(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    tool = create_retriever_tool(
        retriever,
        name="stock_market_search",
        description="Search tool for stock market PDF 2024"
    )

    tools_dict = {tool.name: tool}
    
    llm = LLM().get_llm()
    memory = Memory(llm).get_memory()
    
    rag_agent, memory = GraphBuilder.build(llm, memory, tools_dict, Prompt.system_prompt)

    print("\n=== RAG AGENT READY ===")
    memory_state = memory.load_memory_variables({})["memory"]

    while True:
        user_input = input("\nAsk a question (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = rag_agent.invoke({
            "messages": [HumanMessage(content=user_input)],
            "memory": memory_state
        })

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)
        memory_state = result.get("memory", memory_state)

if __name__ == "__main__":
    main()
