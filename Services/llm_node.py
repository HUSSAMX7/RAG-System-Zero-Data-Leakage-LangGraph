from langchain_core.messages import SystemMessage
from services.agent_state import AgentState


def call_llm(llm, summary_memory):
    """Call LLM and save to memory."""
    
    system_prompt = """
        You are an intelligent AI assistant based on the PDF document loaded into your knowledge base.
        Use the retriever tool available to answer questions. You can make multiple calls if needed.
        If you need to look up some information before asking a follow up question, you are allowed to do that!
        Please always cite the specific parts of the documents you use in your answers.
        """

    def inner(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)]
        messages += state.get("memory", [])  
        messages += state["messages"]        

        response = llm.invoke(messages)

        summary_memory.save_context(
            {"input": state["messages"][-1].content},
            {"output": response.content}
        )

        updated_memory = summary_memory.load_memory_variables({})["history"]

        return {
            "messages": [response],
            "memory": updated_memory  
        }

    return inner