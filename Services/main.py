from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()


from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import MessagesPlaceholder





llm = ChatOpenAI(
    model="gpt-4o", temperature = 0) 

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

summary_memory = ConversationSummaryMemory(
    llm=llm,               
    memory_key="memory",   
    return_messages=True   
)


pdf_path = "C:/Users/hosam/Desktop/ai.pdf"


# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

pages = pdf_loader.load()


# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = "\local"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memory: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools} 

def call_llm(state: AgentState) -> AgentState:
    
    messages = [SystemMessage(content=system_prompt)]
    messages += state.get("memory", [])  
    messages += state["messages"]

    response = llm.invoke(messages)

    
    summary_memory.save_context(
        {"input": messages[-1].content},
        {"output": response.content}
    )

    updated_memory = summary_memory.load_memory_variables({})["memory"]

    return {
        "messages": [response],
        "memory": updated_memory  
    }




def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        tool_name = t['name']
        tool_args = t['args'].get('query', 'No query provided')

        print(f"Calling Tool: {tool_name} with query: {tool_args}")

        if tool_name not in tools_dict:
            print(f"\nTool: {tool_name} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[tool_name].invoke(tool_args)
            print(f"Result length: {len(str(result))}")

        tool_message = ToolMessage(
            tool_call_id=t['id'],
            name=tool_name,
            content=str(result)
        )

        results.append(tool_message)

        summary_memory.save_context(
            {"input": f"[Tool Call] {tool_name}: {tool_args}"},
            {"output": str(result)}
        )

    print("Tools Execution Complete. Back to the model!")

    updated_memory = summary_memory.load_memory_variables({})["memory"]

    return {
        "messages": results,
        "memory": updated_memory
    }


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_ag ent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT with SUMMARY MEMORY ===")

    memory_state = summary_memory.load_memory_variables({})["memory"]

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        result = rag_agent.invoke({
            "messages": [HumanMessage(content=user_input)],
            "memory": memory_state
        })

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

        memory_state = result.get("memory", memory_state)


running_agent()
