import os,tempfile
from typing import Optional
from ava_agent.core.exceptions import SpeechToTextError
from ava_agent.settings import settings
from groq import Groq

class SpeechToText:
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]
    
    def __init__(self):
        """Initialize the SpeechToText class and validate environment variables."""
        # self._validate_env_vars()
        self._client: Optional[Groq] = None

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @property
    def client(self) -> Groq:
        if not self._client:
            self._client=Groq(
                api_key=settings.GROQ_API_KEY
            )
        return self._client

    async def transcribe(self,audio_data: bytes) -> str:
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path=temp_file.name
            try:
                with open(temp_file,"rb") as audio_file:
                    transcription=self.client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3-turbo",
                        language="en",
                        response_format="text"
                    )
                if not transcription:
                    raise SpeechToTextError("Transaction result is empty")
                return transcription
            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            raise SpeechToTextError("Speech to text conversion failed: {}".format(str(e))) from e
    