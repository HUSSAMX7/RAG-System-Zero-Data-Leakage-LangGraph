from langgraph.graph import StateGraph, END
from services.agent_state import AgentState
from services.llm_node import call_llm
from services.tool_node import take_action
from services.should_continue import should_continue
from langchain_core.runnables import Runnable

def build_graph(llm, tools_dict, summary_memory) -> Runnable:
    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm(llm, summary_memory))  
    graph.add_node("retriever_agent", take_action(tools_dict))  

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    return graph.compile()



