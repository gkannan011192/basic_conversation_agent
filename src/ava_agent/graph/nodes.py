import os
from uuid import uuid4
from langchain_core.messages import AIMessage,HumanMessage,RemoveMessage
from langchain_core.runnables import RunnableConfig
from ava_agent.graph.state import AICompanionState
from ava_agent.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from ava_agent.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ava_agent.modules.memory.long_term.memory_manager import get_memory_manager
from ava_agent.modules.schedules.context_generation import ScheduleContextGenerator
from ava_agent.settings import settings

async def router_node(state: AICompanionState):
    chain=get_router_chain()
    response=await chain.ainvoke({"messages":state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    return {"workflow": response.response_type}
    

def context_injection_node(state: AICompanionState):
    """
    Updates the activity context state based on the current schedule.
    This function checks if there's a change in the current activity by comparing
    the schedule context with the stored state. It determines whether a new activity
    should be applied.
    Args:
        state (AICompanionState): The current state object containing companion's context
    Returns:
        dict: A dictionary containing:
            - apply_activity (bool): Flag indicating if a new activity should be applied
            - current_activity (str): The current activity from schedule context
    """

    schedule_context=ScheduleContextGenerator.get_current_activity()
    if schedule_context!=state.get("current_activity",""):
        apply_activity=True
    else:
        apply_activity=False
    return {"apply_activity":apply_activity, "current_activity": schedule_context}

async def conversation_node(state: AICompanionState, config: RunnableConfig):
    """
    Asynchronously processes a conversation node in the AI companion system.
    This function handles the conversation flow by generating contextual responses based on
    current activity, memory context, and previous messages. It utilizes a character response
    chain to generate appropriate AI responses.
    Args:
        state (AICompanionState): The current state of the AI companion containing messages,
            summary, and memory context.
        config (RunnableConfig): Configuration settings for the response generation.
    Returns:
        dict: A dictionary containing an AIMessage with the generated response content.
            Format: {"messages": AIMessage(content=response)}
    Example:
        response = await conversation_node(state, config)
        # Returns {"messages": AIMessage(content="Generated response text")}
    """

    current_activity=ScheduleContextGenerator.get_current_activity()
    memory_context=state.get("memory_context","")
    response_chain=get_character_response_chain(state.get("summary",""))
    response=await response_chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config
    )
    return {"messages": AIMessage(content=response)}

async def image_node(state:AICompanionState, config: RunnableConfig):
    """
    Asynchronously generates and processes images based on conversation context and returns a response.
    This function handles the creation of AI-generated images and corresponding responses within
    a conversation context. It uses the current state and activity information to generate
    appropriate scenarios and images.
    Parameters:
        state (AICompanionState): The current state object containing conversation history 
            and other contextual information
        config (RunnableConfig): Configuration parameters for the running environment
    Returns:
        dict: A dictionary containing:
            - messages (AIMessage): The AI's response to the generated scenario
            - image_path (str): The file path to the generated image
    The function performs the following steps:
    1. Retrieves current activity and memory context
    2. Creates a scenario based on recent messages
    3. Generates an image based on the scenario
    4. Creates a response incorporating the generated image
    5. Returns both the response and the image path
    Raises:
        Any exceptions from the underlying image generation or chain invocation
        processes may be propagated
    """
    
    current_activity= ScheduleContextGenerator.get_current_activity()
    memory_context=state.get("memory_context","")
    chain=get_character_response_chain(state.get("summary"),"")
    text_to_image_module=get_text_to_image_module()
    scenario=await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images",exist_ok=True)
    image_path="generated_images/image_{}.png".format(str(uuid4()))
    await text_to_image_module.generate_image(scenario.image_prompt,image_path)
    scenario_message=HumanMessage(
        content="<image attached by the agent generated from the prompt: {}>".format(scenario.image_prompt)
    )
    updated_messages=state["messages"]+[scenario_message]
    response=await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config
    )
    return {"messages": AIMessage(content=response),"image_path": image_path}

async def audio_node(state: AICompanionState, config: RunnableConfig):
    current_activity= ScheduleContextGenerator.get_current_activity()
    memory_context=state.get("memory_context","")
    chain=get_character_response_chain(state.get("summary",""))
    text_to_speech_module=get_text_to_speech_module()
    response=await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config
    ) 
    output_audio=await text_to_speech_module.synthesize(response)
    return {"messages": response, "audio_buffer": output_audio}

async def summarize_conversation_node(state: AICompanionState):
    """
    Asynchronously summarizes the conversation history and manages message retention.
    This function either creates a new summary of the entire conversation or extends an existing
    summary based on recent messages. It also handles cleanup of old messages to maintain
    conversation history within limits.
    Args:
        state (AICompanionState): The current state of the AI companion containing conversation
            history and other relevant information.
    Returns:
        dict: A dictionary containing:
            - 'summary' (str): The new or updated conversation summary
            - 'messages' (list): List of RemoveMessage objects for cleanup of old messages
    Example:
        result = await summarize_conversation_module(state)
        new_summary = result['summary']
        messages_to_delete = result['messages']
    Note:
        The function uses settings.TOTAL_MESSAGES_AFTER_SUMMARY to determine how many
        recent messages to retain after summarization.
    """

    model=get_chat_model()
    summary=state.get("summary","")
    if summary:
        summary_message=(
            "Summary of the conversation to date between agent and the user: {}\n\n Extend the summary by taking into account the messages above".format(summary)
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between agent and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between agent and the user:"
        )
    messages=state["messages"]+HumanMessage(content=summary_message)
    response=await model.ainvoke(messages)
    delete_messages=[RemoveMessage(id=message.id) for message in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content,"messages": delete_messages}

async def memory_extraction_node(state: AICompanionState):
    if not state["messages"]:
        return {}
    memory_manager=get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}

def memory_injection_node(state: AICompanionState):
    """
    Injects relevant memories into the AI companion's state based on recent conversation context.
    This function retrieves relevant memories from the memory manager based on the last 5 messages
    in the conversation and formats them for use in the prompt.
    Args:
        state (AICompanionState): The current state of the AI companion containing conversation history.
    Returns:
        dict: A dictionary containing the formatted memory context under the key 'memory_context'.
    """

    memory_manager=get_memory_manager()
    recent_context=" ".join([message.content for message in state["messages"][-5:]])
    memories=memory_manager.get_relevent_memories(recent_context)
    memory_context=memory_manager.format_memories_for_prompt(memories)
    return {"memory_context": memory_context}


