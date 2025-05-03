import os,tempfile
from typing import Optional
from ava_agent.core.exceptions import TextToSpeechError
from ava_agent.settings import settings
from elevenlabs import ElevenLabs,Voice,VoiceSettings

class TextToSpeech:
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(self):
        """
        Initialize the TextToSpeech class.
        This constructor initializes a TextToSpeech instance by validating required
        environment variables and setting up an ElevenLabs client connection.
        Attributes:
            _client (Optional[ElevenLabs]): ElevenLabs client instance, initially set to None.
        """
        # self._validate_env_vars()
        self._client:Optional[ElevenLabs]=None

    def _validate_env_vars(self) -> None:
        """
        Validates the required environment variables are present in the system.
        Checks if all environment variables defined in `REQUIRED_ENV_VARS` are set.
        If any required variables are missing, raises a ValueError with the list of missing variables.
        Raises:
            ValueError: If one or more required environment variables are not set.
        """
        missing_vars=[var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError("Missing required environment variables: {}".format(",".join(missing_vars)))
    
    @property
    def client(self) ->ElevenLabs:
        if not self._client:
            self._client=ElevenLabs(
                api_key=settings.ELEVENLABS_API_KEY
            )
        return self._client
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesizes text to speech using ElevenLabs API.
        This method converts the input text to speech audio bytes using the configured
        ElevenLabs voice model and settings.
        Args:
            text (str): The input text to convert to speech. Must not be empty and 
                       should not exceed 5000 characters.
        Returns:
            bytes: The generated audio as bytes.
        Raises:
            ValueError: If the input text is empty or exceeds 5000 characters.
            TextToSpeechError: If audio generation fails or produces empty output.
        """

        if not text.strip():
            raise ValueError("Input text cannnot be empty")
        if len(text)> 5000:
            raise ValueError("Input text cannot exceed 5000 chars")
        
        try:
            audio_generator=self.client.generate(
                text=text,
                voice=Voice(
                    voice_id=settings.ELEVENLABS_VOICE_ID,
                    settings=VoiceSettings(stability=0.5, similarity_boost=0.5),
                ),
                model=settings.TTS_MODEL_NAME
            )
            audio_bytes=b"".join(audio_generator)
            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")
            return audio_bytes
        except Exception as e:
            raise TextToSpeechError("Text to speech conversion for the text: {} failed".format(text))
