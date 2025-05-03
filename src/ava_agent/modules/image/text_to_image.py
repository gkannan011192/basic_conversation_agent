import base64
import logging
import os
from typing import Optional

from ava_agent.core.exceptions import TextToImageError
from ava_agent.core.prompts import IMAGE_ENHANCEMENT_PROMPT,IMAGE_SCENARIO_PROMPT
from ava_agent.settings import settings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from together import Together

class ScenarioPrompt(BaseModel):
    """A data model representing a scenario with narrative and image generation components.
    This class inherits from BaseModel and encapsulates both the narrative response
    and the corresponding image generation prompt for a given scenario.
    Attributes:
        narrative (str): The AI's narrative/textual response to a given question or prompt
        image_prompt (str): A carefully crafted prompt used to generate an image that 
            visually represents the scenario described in the narrative
    Examples:
        scenario = ScenarioPrompt(
            narrative="A sunny day in Paris with the Eiffel Tower in view",
            image_prompt="Photorealistic view of Eiffel Tower on a bright sunny day, 
                         blue skies with scattered clouds"
        )
    """

    narrative: str=Field(..., description="The AI's narrative response to the question")
    image_prompt: str=Field(..., description="The visual prompt to generate an image representing the image")

class EnhancedPrompt(BaseModel):
    """A data model representing an enhanced text prompt for image generation.
    This class extends BaseModel to provide a structured way of handling 
    enhanced text prompts used in image generation tasks.
    Attributes:
        content (str): The enhanced textual prompt used to generate an image.
                      This is a required field that contains the detailed 
                      description or instructions for image generation.
    """

    content: str=Field(
        ...,
        description="The enhanced text prompt to generate the image"
    )

class TextToImage:
    
    REQUIRED_ENV_VARS=["GROQ_API_KEY","TOGETHER_API_KEY"]
    
    def __init__(self):
        # self._validate_env_vars()
        self._together_client: Optional[Together]=None
        self.logger=logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """
        Validates that all required environment variables are present.
        This method checks if all environment variables specified in REQUIRED_ENV_VARS
        are defined in the current environment.
        Raises:
            ValueError: If any required environment variables are missing,
                       with a message listing the missing variables.
        """
        
        missing_vars=[var for var in self.REQUIRED_ENV_VARS if not os.environ.get(var)]
        if missing_vars:
            raise ValueError("\nMissing required environment variables: {}".format(' ,').join(missing_vars))
        
    @property
    def together_client(self) -> Together:
        """
        Property that lazily initializes and returns a Together client instance.
        This property manages a singleton pattern for the Together client, creating it only
        when first accessed and reusing the same instance for subsequent calls.
        Returns:
            Together: An initialized Together client instance using the API key from settings.
        Note:
            The Together client is instantiated with the TOGETHER_API_KEY from the application settings.
        """

        if not self._together_client:
            self._together_client=Together(api_key=settings.TOGETHER_API_KEY)
        return self._together_client

    async def generate_image(self,prompt: str, output_path: str="") -> bytes:
        """
        Generates an image based on the given text prompt using Together AI's image generation service.
        Args:
            prompt (str): The text description used to generate the image.
            output_path (str, optional): Path where the generated image should be saved. 
                                        If empty, image won't be saved to disk. Defaults to "".
        Returns:
            bytes: The generated image data as base64 encoded bytes.
        Raises:
            ValueError: If the prompt is empty or contains only whitespace.
            TextToImageError: If image generation fails for any reason.
        Example:
            ```
            text_to_image = TextToImage()
            image_data = await text_to_image.generate_image(
                prompt="A beautiful sunset over mountains",
                output_path="images/sunset.png"
            ```
        """
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        try:
            self.logger.info("Generating image for the prompt: {}".format(prompt))
            together_response=self.together_client.images.generate(
                prompt=prompt,
                model=settings.TTI_MODEL_NAME,
                width=1024,
                height=800,
                steps=4,
                n=1,
                response_format="b64_json"
            )
            image_data = base64.b64encode(together_response.data[0].b64_json)

            if output_path and image_data:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path,"wb") as f:
                    f.write(image_data)
                self.logger.info("Image saved to the file: {}".format(output_path))

        except Exception as e:
            raise TextToImageError("Failed to create image for the prompt: {}".format(prompt))
    
    async def create_scenario(self, chat_history: list=None) -> ScenarioPrompt:
        """
        Creates a scenario prompt based on chat history using LLM.
        This method processes recent chat history to generate a structured scenario prompt using
        the Groq LLM model. It formats the chat history and uses a language model to generate
        a scenario that follows the ScenarioPrompt structure.
        Args:
            chat_history (list, optional): List of chat messages containing type and content.
                Defaults to None. Only the 5 most recent messages are considered.
        Returns:
            ScenarioPrompt: A structured scenario prompt generated from the chat history.
        Raises:
            TextToImageError: If scenario creation fails for any reason.
        Example:
            >>> chat_hist = [Message(type="user", content="Create an image of a sunset")]
            >>> scenario = await image_handler.create_scenario(chat_hist)
        """
        
        try:
            formated_history = "\n".join("{}.{}".format(msg.type.title(),msg.content) for msg in chat_history[0:5])
            self.logger.info("Creating scenario from the chat history: {}".format(formated_history))
            llm_client=ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.4,
                max_retries=3
            )

            structured_llm_client = llm_client.with_structured_output(ScenarioPrompt)
            llm_chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT
                )
                | structured_llm_client
            )

            new_scenario= llm_chain.invoke({"chat_history": formated_history})
            self.logger.info("Created Scenario :{}".format(new_scenario))
            return new_scenario

        except Exception as e:
            raise TextToImageError("Failed to create scenario : {}".format(str(e))) from e
    
    async def enhance_prompt(self, prompt: str) -> str:
        """
        Enhances the given text prompt using a language model to create more detailed and effective image generation prompts.
        Args:
            prompt (str): The original text prompt to be enhanced.
        Returns:
            str: The enhanced version of the input prompt optimized for image generation.
        Raises:
            TextToImageError: If prompt enhancement fails due to any errors in the process.
        Note:
            This method uses the Groq language model specified in settings to improve prompt quality
            by adding more descriptive details and artistic direction.
        """

        try:
            self.logger.info("Enhancing the prompt: {}".format(prompt))

            llm_client=ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.25,
                max_retries=3
            )
            
            structured_llm_client = llm_client.with_structured_output(EnhancedPrompt)
            llm_chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT
                )
                | structured_llm_client
            )
            enhanced_prompt = llm_chain.invoke({"prompt":prompt}).content
            self.logger.info("Enhanced prompt: {}".format(enhanced_prompt))
            return enhanced_prompt
        except Exception as e:
            raise TextToImageError("Failed to enhance the prompt: {}".format(enhanced_prompt))