from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


class Memory:
    def __init__(self, llm):
        self.memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="memory",
            return_messages=True
        )

    def get_memory(self):
        return self.memory
