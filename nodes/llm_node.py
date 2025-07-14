
from langchain.memory import ConversationSummaryMemory


class llm_node:
        
    def call_llm(llm, summary_memory, system_prompt, tools_dict):

        def inner(state):
            messages = [SystemMessage(content=system_prompt)]
            messages += state.get("memory", [])
            messages += state["messages"]

            response = llm.invoke(messages)


            summary_memory.save_context(
                {"input":messages[-1].content},
                {"output": response.content}
            )

            updated_memory = summary_memory.load_memory_variables({})["memory"]

            return { 
                "messages":[response],
                "memory":updated_memory
            }
        
        return inner