from operator import itemgetter
from typing import List, Tuple

from langchain_community.llms import LlamaCpp
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompt_values import StringPromptValue
from langchain_core.messages import ChatMessage

from langchain.globals import set_verbose, set_debug

# set_debug(True)

SYSTEM_PROMPT ="""You are a Dungeon Master for a D&D Forgotten Realms campaign set in Waterdeep. 
The following is a conversation between you and the player. 
Reply only for yourself, and directly to the player.
Keep the conversation natural, and use your own personality."""

USER_PROMPT_TEMPLATE = """{input}"""

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    model = LlamaCpp(
            model_path="models/neural-chat-7b-v3-3.Q5_K_M.gguf",
            temperature=0.1,
            max_tokens=512,
            top_p=0.95,
            # callback_manager=callback_manager,
            verbose=True, # Verbose is required to pass to the callback manager
            n_gpu_layers=-1,
            echo=True,
            n_ctx=1024*32,
        )
    prompt = ChatPromptTemplate.from_messages([
        ChatMessagePromptTemplate.from_template(SYSTEM_PROMPT, role="### System"),
        MessagesPlaceholder("chat_history"),
        ChatMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE, role="### User"),
        ChatMessagePromptTemplate.from_template("", role="### Assistant")]
    )
    
    
    memory = ConversationBufferWindowMemory(
        k=2, 
        return_messages=True, 
        input_key="input", 
        memory_key="history",
        output_key="answer",
        )

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    # logger.debug(f"memory:load_memory_variables: {memory.chat_memory}")

    # standalone_question = {
    #     "standalone_question": {
    #         "input": lambda x: x["input"],
    #         "history": lambda x: x["chat_history"],
    #     } | prompt
    # }
    
    final_chain = loaded_memory | prompt | model | StrOutputParser()
    cl.user_session.set("runnable", final_chain)
    cl.user_session.set("chat_history", memory)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    memory: ConversationBufferWindowMemory = cl.user_session.get("chat_history")
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    memory.chat_memory.add_message(ChatMessage(content=message.content, 
                                            role="### User"))
    memory.chat_memory.add_message(ChatMessage(content=msg.content, 
                                               role="### Assistant"))
    
# https://python.langchain.com/docs/expression_language/how_to/message_history
# https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
# https://docs.chainlit.io/integrations/langchain
# https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py
# https://python.langchain.com/docs/integrations/llms/llamacpp
# https://github.com/langchain-ai/langchain/discussions/13674