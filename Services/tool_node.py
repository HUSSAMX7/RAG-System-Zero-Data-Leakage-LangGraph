from langchain_core.messages import ToolMessage
from services.agent_state import AgentState




def take_action(tools_dict): 
    def inner(state: AgentState) -> AgentState:  
        tool_calls = state['messages'][-1].tool_calls  
        results = []
        for t in tool_calls:
            query = t['args'].get('query', '')
            if t['name'] not in tools_dict:
                result = "Tool not found."
            else:
                result = tools_dict[t['name']].invoke(query)
            
            results.append(
                ToolMessage(
                    tool_call_id=t['id'],
                    name=t['name'],
                    content=str(result)
                )
            )

        return {'messages': results}
    return inner
