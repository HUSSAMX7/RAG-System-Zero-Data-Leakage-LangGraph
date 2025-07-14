

class too_node:
        
    def take_action(tools_dict, summary_memory):
        def inner(state):
            tool_calls = state['messages'][-1].tool_calls
            results = []

            for t in tool_calls:
                name = t['name']
                query = t['args'].get("query", "")

                if name in tools_dict:
                    output = tools_dict[name].invoke(query)
                else:
                    output = f"Tool '{name}' not found."

                tool_message = ToolMessage(
                    tool_call_id=t['id'],
                    name=name,
                    content=output
                )
                results.append(tool_message)

                summary_memory.save_context(
                    {"input": f"[Tool Call] {name}: {query}"},
                    {"output": output}
                )

            updated_memory = summary_memory.load_memory_variables({})["memory"]
            return {
                "messages": results,
                "memory": updated_memory
            }

        return inner
