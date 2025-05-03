import logging
from typing import Dict, List, Optional
from pathlib import Path

# Core imports
from ava_agent.settings import settings

# Module imports
from ava_agent.modules.image import TextToImage
from ava_agent.modules.speech import TextToSpeech
import os 

# Graph imports
from ava_agent.graph import graph_builder

# # Configure logging
# logging.basicConfig(
#     level=settings.LOG_LEVEL,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

text_to_image = TextToImage()
text_to_speech = TextToSpeech()

if __name__=="__main__":
    print("Check method")
    # REQUIRED_ENV_VARS=["GROQ_API_KEY","TOGETHER_API_KEY"]
    # print(os.environ.get("GROQ_API_KEY"))

