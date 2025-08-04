from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationSummaryMemory

load_dotenv()

from services.llm import get_llm
from services.embeddings import get_embeddings
from services.load import DocumentLoader
from services.text_splitter import split_documents
from services.vectorstore import VectorStore
from services.tool_retriever import create_retriever_tool
from services.build_graph import build_graph

llm = get_llm()

summary_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)

embeddings = get_embeddings()
documents = DocumentLoader.load_from_pdf("C:/Users/hosam/OneDrive/سطح المكتب/ai_eng.pdf")
chunks = split_documents(documents)
retriever = VectorStore.build_vectorstore(chunks, embeddings)

tool = create_retriever_tool(retriever)
llm = llm.bind_tools([tool])
tools_dict = {tool.name: tool}

rag_agent = build_graph(llm, tools_dict, summary_memory)

chat_memory = []

def running_agent():
    global chat_memory
    print("\n=== RAG AGENT (type 'exit' to quit) ===")

    while True:
        user_input = input("\nWhat is your question: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        state = {
            "messages": [HumanMessage(content=user_input)],
            "memory": chat_memory
        }

        try:
            result = rag_agent.invoke(state)

            print("\n--- Answer ---")
            print(result["messages"][-1].content)

            chat_memory = result.get("memory", chat_memory)

        except Exception as e:
            print("\n[Error occurred]:", e)

running_agent()