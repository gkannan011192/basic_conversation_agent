import logging
import uuid
from datetime import datetime
from typing import List, Optional
from ava_agent.core.prompts import MEMORY_ANALYSIS_PROMPT
from ava_agent.modules.memory.long_term.vector_store import get_vector_store
from ava_agent.settings import settings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field

class MemoryAnalysis(BaseModel):
    """A data model for analyzing and formatting memories.
    This class represents the analysis of a memory, determining its importance and providing
    a formatted version for storage.
    Attributes:
        is_important (bool): Flag indicating if the message is significant enough to be stored
            in the memory system.
        formatted_memory (Optional[str]): The processed and formatted version of the memory
            ready for storage. Can be None if memory is not important.
    """

    is_important: bool=Field(
        ...,
        description="Whether the message is important enough to be stored in the memory",
    )
    formatted_memory: Optional[str]=Field(
        ...,
        description="The formatted memory to be stored"
    )

class MemoryManager:
    def __init__(self):
        self.vector_store=get_vector_store()
        self.logger=logging.getLogger(__name__)
        self.llm_client=ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_retries=3
        ).with_structured_output(MemoryAnalysis)
    
    async def _analyze_memory(self,message:str) -> MemoryAnalysis:
        """
        Analyzes a given message to determine memory-related characteristics.
        This method performs memory analysis on the input message using a predefined prompt
        template and the LLM client.
        Args:
            message (str): The message text to analyze.
        Returns:
            MemoryAnalysis: Analysis results containing memory-related characteristics of the message.
        Raises:
            Any exceptions raised by the LLM client during invocation.
        """
        
        prompt=MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm_client.ainvoke(prompt)

    async def extract_and_store_memories(self,message: BaseMessage) -> None:
        """
        Analyzes and stores important memories from user messages in the vector store.
        This method processes incoming messages to extract potentially important memories,
        checks for duplicates, and stores new memories with metadata if they are deemed important.
        Args:
            message (BaseMessage): The message object containing the content to analyze.
                Must have 'type' and 'content' attributes.
        Returns:
            None
        Notes:
            - Only processes messages of type "human"
            - Skips storage if similar memory already exists in vector store
            - Stores new memories with UUID and timestamp metadata
            - Uses internal _analyze_memory method to evaluate importance
        """

        if message.type!="human":
            return
        memory_analysis=await self._analyze_memory(message.content)
        print("Memory analysis: {}".format(memory_analysis))
        if memory_analysis and  memory_analysis.is_important and memory_analysis.formatted_memory:
            similar_memory=self.vector_store.find_similar_memory(memory_analysis.formatted_memory)
            if similar_memory:
                self.logger.info(
                    "Similar memory that already exists: {}".format(memory_analysis.formatted_memory)
                )
                return
            self.logger.info("Storing new memory : {}".format(memory_analysis.formatted_memory))
            self.vector_store.store_memory(
                text=memory_analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )
    
    def get_relevent_memories(self,context: str) -> List[str]:
        """
        Retrieves relevant memories based on the given context.
        This method searches the vector store for memories that are semantically similar to the provided context
        and returns the top-k most relevant memories based on the MEMORY_TOP_K setting.
        Args:
            context (str): The context string to search for relevant memories.
        Returns:
            List[str]: A list of memory texts that are most relevant to the given context.
                       Returns an empty list if no relevant memories are found.
        Note:
            - The method logs debug information about each found memory and its relevance score.
            - The number of returned memories is limited by settings.MEMORY_TOP_K.
        """

        memories=self.vector_store.search_memories(context,k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
            return [memory.text for memory in memories]

    def format_memories_for_prompt(self,memories: List[str]) -> str:
        """
        Formats a list of memories into a string suitable for inclusion in a prompt.
        The function takes a list of memories and formats them as a bullet-pointed list,
        with each memory preceded by a dash (-). If the input list is empty, returns an empty string.
        Args:
            memories (List[str]): A list of memory strings to be formatted
        Returns:
            str: A formatted string where each memory is on a new line starting with "- ",
                 or an empty string if the input list is empty.
        Example:
            >>> format_memories_for_prompt(["memory1", "memory2"])
            "- memory1\n- memory2"
            >>> format_memories_for_prompt([])
            ""
        """

        if not memories:
            return ""
        return "\n".join("- {}".format(memory) for memory in memories)
    
def get_memory_manager() -> MemoryManager:
    return MemoryManager()