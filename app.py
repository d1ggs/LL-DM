import chainlit as cl
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
# from langchain.schema import StrOutputParser
# from langchain_community.llms import LlamaCpp
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent import GuidanceDMAgent
# set_debug(True)

# SYSTEM_PROMPT ="""You are a Dungeon Master for a D&D Forgotten Realms campaign set in Waterdeep. 
# The following is a conversation between you and the player. 
# Reply only for yourself, and directly to the player.
# Keep the conversation natural, and use your own personality."""

SYSTEM_PROMPT = """
You are a dungeon master of the D&D 5th edition game. 
You are supposed to run a campaign for a single player using the official rules of D&D. 
You know every rule and your role is to guide the player, describe the scenes, answer questions, and tell the story.
Follow the following mandatory guidelines:
* always follow the rules of D&D 5th edition
* do not deliberately favor the players. Always rely on dice rolls for determining the outcomes of events that require dice rolls
* always ask the player for their next action. Do not assume their next action

Your messages must be structured as follows:
<description of the scene / answer to the player's request>
<ask for the next player action / input>
"""

USER_PROMPT_TEMPLATE = """{question}"""

@cl.cache
def load_llm():
    
    llm = GuidanceDMAgent("/mnt/d/wsl/llamacpp/models")
    # model = LlamaCpp(
    #     model_path="models/neural-chat-7b-v3-3.Q5_K_M.gguf",
    #     temperature=0.1,
    #     max_tokens=512,
    #     top_p=0.95,
    #     # callback_manager=callback_manager,
    #     verbose=True, # Verbose is required to pass to the callback manager
    #     n_gpu_layers=-1,
    #     echo=True,
    #     n_ctx=1024*32,
    #     stop=["### System", "### User", "### Assistant", "User:", "Player:", "DM:"],
    # )
    # return model
    
    return llm

# from transformers import AutoProcessor, SeamlessM4Tv2Model
# import torchaudio
# @cl.cache
# def load_translation_model():
#     processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", device_map="cpu")
#     model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large", device_map="cpu")    
#     return model, processor

@cl.on_chat_start
async def on_chat_start():
    
    # translation_model, translation_processor = load_translation_model()
    # cl.user_session.set("translation_model", translation_model)
    # cl.user_session.set("translation_processor", translation_processor)

    # prompt = ChatPromptTemplate.from_messages([
    #     ChatMessagePromptTemplate.from_template(SYSTEM_PROMPT, role="### System"),
    #     MessagesPlaceholder("chat_history"),
    #     ChatMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE, role="### User"),
    #     ChatMessagePromptTemplate.from_template("", role="### Assistant")]
    # )
    
    agent = load_llm()
    memory = []
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("memory", memory)


@cl.on_message
async def on_message(message: cl.Message):
    
    agent: GuidanceDMAgent = cl.user_session.get("agent")
    history: list = cl.user_session.get("memory")

    res = agent.generate_output_no_history(message.content)
    
    history.append({"role": "### User",
                    "message": message})
    history.append({"role": "### System",
                    "message": res})
    
    await cl.Message(content=res).send()
    
    # message_text = message.content
    # translation_model = cl.user_session.get("translation_model")
    # translation_processor = cl.user_session.get("translation_processor")
    # audio_tmp_dir = cl.user_session.get("temp_dir")
    
    # runnable = cl.user_session.get("runnable")  # type: Runnable
    # memory: ConversationBufferWindowMemory = cl.user_session.get("chat_history")
    # elements = []
    # llm_msg = cl.Message(content="", elements=[])

    # async for chunk in runnable.astream(
    #     {"input": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await llm_msg.stream_token(chunk)
    
    # memory.chat_memory.add_message(ChatMessage(content=message.content, 
    #                                         role="### User"))
    # memory.chat_memory.add_message(ChatMessage(content=llm_msg.content, 
    #                                            role="### Assistant"))
    
    # text_inputs = translation_processor(text = llm_msg.content, src_lang="eng", return_tensors="pt") #.to("cuda")
    # audio_tensor = translation_model.generate(**text_inputs, tgt_lang="eng")[0].cpu()
    # # Save the audio to a file with the timestamp as the name
    # now = datetime.now().isoformat()
    # destination = os.path.join(audio_tmp_dir, f"{now}.wav")
    # torchaudio.save(destination, 
    #                 audio_tensor, 
    #                 15000)    
    # elements.append(
    #     cl.Audio(name="Italian audio", path=destination, display="inline"),
    # )
    # await llm_msg.send()
    # llm_msg.elements = elements
    # await llm_msg.update()
    
# https://python.langchain.com/docs/expression_language/how_to/message_history
# https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
# https://docs.chainlit.io/integrations/langchain
# https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py
# https://python.langchain.com/docs/integrations/llms/llamacpp
# https://github.com/langchain-ai/langchain/discussions/13674