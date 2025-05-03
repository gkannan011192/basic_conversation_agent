from langgraph.graph import END
from typing_extensions import Literal
from ava_agent.graph.state import AICompanionState
from ava_agent.settings import settings

def should_summarize_conversation(state: AICompanionState,) -> Literal["summarize_conversation_node","__end__"]:
    """
    Determines if the conversation needs to be summarized based on message count.
    This function checks if the total number of messages in the conversation state
    exceeds a configured threshold and decides whether to trigger conversation
    summarization.
    Args:
        state (AICompanionState): The current state of the AI companion containing
            conversation messages.
    Returns:
        Literal["summarize_conversation_node","__end__"]: Returns "summarize_conversation_node" 
            if messages exceed threshold, otherwise returns "__end__"
    """
    
    messages=state["messages"]
    if len(messages)>settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"
    return END

def select_workflow(state: AICompanionState,) -> Literal["conversation_node","image_node","audio_node"]:
    workflow=state["workflow"]
    if workflow=="image":
        return "image_node"
    elif workflow=="audio":
        return "audio_node"
    else:
        return "conversation_node"
