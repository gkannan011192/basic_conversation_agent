from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from pydantic import BaseModel,Field
from ava_agent.core.prompts import CHARACTER_CARD_PROMPT,ROUTER_PROMPT
from ava_agent.graph.utils.helpers import AsteriskRemoveParser,get_chat_model

class RouteResponse(BaseModel):
    response_type: str=Field(
        description="The response to give to the user. It must one of 'conversation', 'image' or 'audio' "
    )

def get_router_chain():
    """
    Creates and returns a router chain for message routing.
    This function constructs a chain that combines a chat prompt template with a language model
    to route messages to appropriate handlers. The chain uses a structured output format defined
    by RouteResponse.
    Returns:
        Chain: A composed chain consisting of a prompt template piped to a chat model,
              configured to output RouteResponse objects. The chain is used for routing
              decisions based on user messages.
    """

    model=get_chat_model(temperature=0.3).with_structured_output(RouteResponse)
    prompt=ChatPromptTemplate.from_messages(
        [("system",ROUTER_PROMPT),MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model

def get_character_response_chain(summary: str=""):
    """
    Creates a chain for generating character responses in a conversation.
    This function sets up a conversation chain that processes messages through a character-based
    language model, maintaining context and character consistency.
    Parameters:
        summary (str, optional): A summary of the previous conversation context. Defaults to empty string.
    Returns:
        Chain: A language model chain composed of:
            - A chat prompt template with system message and message placeholder
            - A chat language model
            - An asterisk removal parser
    The chain processes messages based on a character card prompt, optionally incorporating
    conversation history through the summary parameter.
    Example:
        >>> chain = get_character_response_chain("User asked about favorite color")
        >>> response = chain.invoke({"messages": messages})
    """

    model=get_chat_model()
    system_message=CHARACTER_CARD_PROMPT
    if summary:
        system_message+="\n\nSummary of the earlier conversation between agent and the user: {}".format(summary)
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | model | AsteriskRemoveParser()