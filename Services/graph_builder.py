class GraphBuilder:
    @staticmethod
    def build(llm, memory, tools_dict, system_prompt):
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Annotated, Sequence
        from operator import add as add_messages
        from langchain_core.messages import BaseMessage

        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            memory: Annotated[Sequence[BaseMessage], add_messages]

        should_continue = lambda state: hasattr(state["messages"][-1], "tool_calls")

        graph = StateGraph(AgentState)
        graph.add_node("llm", llm_node.call_llm(llm, memory, system_prompt, tools_dict))
        graph.add_node("tool", too_node.take_action(tools_dict, memory))
        graph.add_conditional_edges("llm", should_continue, {True: "tool", False: END})
        graph.add_edge("tool", "llm")
        graph.set_entry_point("llm")

        return graph.compile(), memory
