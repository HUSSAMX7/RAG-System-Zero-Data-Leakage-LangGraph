
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memory: Annotated[Sequence[BaseMessage], add_messages]
