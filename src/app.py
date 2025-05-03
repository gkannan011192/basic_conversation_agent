from io import BytesIO
import chainlit as cl
from langchain_core.messages import AIMessageChunk,HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ava_agent.graph import graph_builder
from ava_agent.modules.image.image_to_text import ImageToText
from ava_agent.modules.speech import SpeechToText,TextToSpeech
from ava_agent.settings import settings

speech_to_text=SpeechToText()
text_to_speech= TextToSpeech()
image_to_text=ImageToText()

@cl.on_chat_start
async def on_chat_start():
    """
    Asynchronous function that initializes the chat session.
    This function is called when a new chat session starts. It sets up the initial thread ID
    in the user's session to 1.
    Returns:
        None
    """
    cl.user_session.set("thread_id",1)

@cl.on_message
async def on_message(message: cl.Message):
    """
    Asynchronous message handler for processing user messages in a chat interface.
    This function processes incoming messages, handles different types of content (text, images),
    maintains conversation context, and manages various output workflows (text, audio, image).
    Args:
        message (cl.Message): The incoming message object containing content and possibly elements
                             like images.
    Returns:
        None: Messages are sent asynchronously through the chat interface.
    Flow:
        1. Processes any attached images and adds their descriptions to the message content
        2. Retrieves the conversation thread ID from the user session
        3. Uses a graph-based processing system with short-term memory to generate responses
        4. Handles different output workflows:
            - Audio: Sends response with audio playback
            - Image: Sends response with inline image
            - Text: Streams text response
    Raises:
        No explicit exceptions are raised, but image processing errors are logged as warnings.
    Dependencies:
        - chainlit (cl)
        - AsyncSqliteSaver
        - graph_builder
        - image_to_text analyzer
        - settings configuration
    """
    msg=cl.Message(content="")
    content=message.content
    if message.elements:
        for element in message.elements:
            if isinstance(element,cl.Image):
                with open(element.path,"rb") as fileRead:
                    image_bytes=fileRead.read()
                try:
                    image_description=await image_to_text.analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of the conversation"
                    )
                    content+="\n Image Analysis: {}".format(image_description)
                except Exception as e:
                    cl.logger.warning("Failed to get the image description : {}".format(e))
    
    thread_id=cl.user_session.get("thread_id")

    async with cl.Step(type="run"):
        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
            graph=graph_builder.compile(checkpointer=short_term_memory)
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=content)]},
                {"configurable":{"thread_id": thread_id}},
                stream_mode="messages"
            ):
                if chunk[1]["langgraph_node"]=="conversation_node" and isinstance(chunk[0],AIMessageChunk):
                    await msg.stream_token(chunk[0].content)

            output_state=await graph.aget_state(config={"configurable":{"thread_id":thread_id}})
    
    if output_state.values.get("workflow")=="audio":
        response=output_state.values["messages"][-1].content
        audio_buffer=output_state.values["audio_buffer"]
        output_audio_element=cl.Audio(
            name="Audio",
            auto_play=True,
            mime="audio/mpeg3",
            content=audio_buffer,
        )
        await cl.Message(content=response,elements=[output_audio_element]).send()
    elif output_state.values.get("workflow")=="image":
        response=output_state.values["messages"][-1].content
        image=cl.Image(path=output_state.values["image_path"],display="inline")
        await cl.Message(content=response,elements=[image]).send()
    else:
        await msg.send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    if chunk.isStart:
        buffer=BytesIO()
        buffer.name="input_audio.{}".format(chunk.mimeType.split('/')[1])
        cl.user_session.set("audio_buffer",buffer)
        cl.user_session.set("audio_mime_type",chunk.mimeType)
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements):
    audio_buffer=cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_data=audio_buffer.read()

    input_audio_element=cl.Audio(mime="audio/mpeg3",content=audio_data)
    await cl.Message(author="You",content="",elements=[input_audio_element,*elements]).send()

    transcription = await speech_to_text.transcribe(audio_data)
    thread_id=cl.user_session.get("thread_id")

    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        llm_graph=graph_builder.compile(checkpointer=short_term_memory)
        output_state=await llm_graph.ainvoke(
            {"messages":[HumanMessage(content=transcription)]},
            {"configurable":{"thread_id": thread_id}}
        )
    
    audio_buffer=await text_to_speech.synthesize(output_state["messages"][-1].content)
    output_audio_ele=cl.Audio(
        name="Audio",
        auto_play=True,
        mime="audio/mpeg3",
        content=audio_buffer
    )
    await cl.Message(
        content=output_state["messages"][-1].content,
        elements=[output_audio_ele]
    ).send()