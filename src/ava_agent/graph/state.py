from langgraph.graph import MessagesState

class AICompanionState(MessagesState):
    """
    AICompanionState is a data structure that represents the state of an AI companion,
    inheriting from the MessagesState class. It encapsulates various attributes related
    to the AI companion's current context, activities, and resources.
    Attributes:
        summary (str): A brief summary or description of the current state.
        workflow (str): The workflow or process the AI companion is currently engaged in.
        audio_buffer (bytes): A buffer containing audio data relevant to the current state.
        image_path (str): The file path to an image associated with the current state.
        current_activity (str): The name or description of the activity the AI companion is performing.
        apply_activity (bool): A flag indicating whether the current activity should be applied or executed.
        memory_context (str): Contextual information or memory relevant to the current state.
    """
    summary: str
    workflow: str
    audio_buffer: bytes
    image_path: str
    current_activity: str
    apply_activity: bool
    memory_context: str
