import base64,logging,os
from typing import Optional,Union
from ava_agent.core.exceptions import ImageToTextError
from ava_agent.settings import settings
from groq import Groq

class ImageToText:
    """
    A class to handle imgae to text capabilities
    """
    REQUIRED_ENV_VARIABLE=["GROQ_API_KEY"]

    def __init__(self):
        # self._validate_env_vars()
        self._client: Optional[Groq ]=None
        self.logger=logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @property
    def client(self) -> Groq:
        """
        Lazy initialization of the Groq client instance.
        Returns:
            Groq: An initialized Groq client object using the API key from settings.
        Note:
            The client is initialized only once on first access and cached for subsequent calls.
        """
        if self._client is None:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    async def analyze_image(self,image_data: Union[str,bytes], prompt: str="") -> str:
        """
        Analyzes an image and generates a text description based on the given prompt.
        This method processes both image file paths and binary image data, converts the image
        to base64 format, and uses OpenAI's Vision API to generate a textual description.
        Args:
            image_data (Union[str, bytes]): Either a file path to an image or binary image data.
            prompt (str, optional): Custom prompt to guide the image analysis. 
                Defaults to "Please describe what you see in the image".
        Returns:
            str: A text description of the image contents.
        Raises:
            ImageToTextError: If there's any error during image analysis, including:
                - File not found
                - Empty image data
                - API response errors
                - Other processing errors
            ValueError: If the image file path doesn't exist or if image data is empty.
        Example:
            >>> analyzer = ImageToText()
            >>> description = await analyzer.analyze_image("path/to/image.jpg")
            >>> description = await analyzer.analyze_image(image_bytes, "Describe the objects in this image")
        """

        try:
            # Handle file path
            if isinstance(image_data,str):
                if not os.path.exists(image_data):
                    raise ValueError("Image file: {} not found".format(image_data))
                with open(image_data,'rb') as f:
                    image_bytes=f.read()
            else:
                image_bytes=image_data
            if not image_bytes:
                raise ValueError("Image data cannot be empty")

            #convert image to base64 format
            base64_image=base64.b64encode(image_bytes).decode("utf-8")

            if not prompt:
                prompt="Please describe what you see in the image"
            
            #Create message for vision api
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type":"image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                }
            ]

            response=self.client.chat.completions.create(
                model=settings.ITT_MODEL_NAME,
                messages=messages,
                max_tokens=1000,
            )

            if not response.choices:
                raise ImageToTextError("No response received from the Vision model")
            
            description=response.choices[0].message.content
            self.logger.info("Generated image description : {}".format(description))
            return description
        
        except Exception as e:
            raise ImageToTextError("Failed to analyze image: {}".format(str(e))) from e


           
            
        