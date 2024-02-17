import os
from enum import Enum
from typing import List, Optional

from guidance import gen, select
from guidance.models import LlamaCppChat

# from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
from loguru import logger

# from llama_index.llms import OpenAILike
from src.chat.history import ChatRole, HistoryMessage
from src.chat.roles import RoleTokens
from src.vector_store import AutoMergingSRDIndex, SRDConfig


class Model(Enum):
    LLAMA_CPP = 0
    OPENAI_API = 1


class ModelsRoleTokens:
    intel_neural_chat = RoleTokens(
        system="### System", human="### User", assistant="### Assistant"
    )


class GuidanceDMAgent:
    """DM Agent based on Guidance library"""

    def __init__(self, model_path: str, srd_config: Optional[SRDConfig] = None):
        self.model_path = model_path
        logger.debug("Loading SRD index")
        self.llama_index_model = self.load_llama_model()
        self.srd = self.load_srd_index(
            srd_config if srd_config is not None else SRDConfig()
        )
        self.lm = LlamaCppChat(self.llama_index_model._model)
        
        self.history: List[HistoryMessage] = []  # Initialize the history list


    def load_openai_compatible_api(self):
        raise NotImplementedError

    def load_llama_model(self):
        logger.debug("Loading LLaMa Model")
        model = LlamaCPP(
            model_path=os.path.join(self.model_path, "neural-chat-7b-v3-3.Q5_K_M.gguf"),
            temperature=0.1,
            max_new_tokens=512,
            # callback_manager=callback_manager,
            # verbose=True, # Verbose is required to pass to the callback manager
            # echo=True,
            context_window=1024 * 32,
            model_kwargs={"n_gpu_layers": -1},
        )
        logger.debug("Model ready")
        return model

    def load_srd_index(self, srd_config: SRDConfig):
        """Loads the Standard Reference Document (SRD) index"""
        # fake_openai = OpenAILike(api_base="http://localhost:5000/v1", api_key="...")
        index = AutoMergingSRDIndex(self.llama_index_model, srd_config)
        return index

    def decide_rules_tool(self, llm, message):
        """Decide whether to use a tool to answer the user message"""
        return llm + (
            f"### System: You are a dungeon master for D&D 5e games. You have the following tools available: ('rules-database', Answers questions about the rules and details of the game). Do not rely on your own knowledge, use the tools to answer the question, but do not use tools when not needed.\n"
            f"### User: {message}\n"
            f"### Assistant: I {select(['need', 'do not need'], name='choice')} to use a tool to answer your question."
        )

    def generate_tool_output(self, tool_decision_output):
        """Based on the decision on whether to use a tool,
        generate the output accordingly."""

        if tool_decision_output["choice"] == "need":
            # Generate the question to be fed to the tool
            question_query = (
                tool_decision_output
                + f"\n### Assistant: I will use the 'rules-database' tool to answer your question. The reworded question, wrapped in <question></question> to be fed to the tool is <question>{gen(name='query', stop='</question>')}'."
            )
            # Use the tool to answer the question
            tool_answer = self.srd.query(question_query["query"]).response

            answer = (
                tool_decision_output
                + f"### Assistant: I will use the 'rules-database' tool to answer your question. The tool's answer is {tool_answer}. \n\nMy final answer is then: {gen(stop='### User:', name='answer')}"
            )
        else:
            answer = (
                tool_decision_output
                + f'### Assistant: I do not need to use a tool to answer your question. {gen(stop="### User:", name="answer")}'
            )

        return answer["answer"]

    def generate_output_no_history(self, message) -> str:
        """Generate the output for the user message with no history"""
        tool_decision_output = self.decide_rules_tool(self.lm, message)
        return self.generate_tool_output(tool_decision_output)

    def generate_chat_history_string(self, history: list[HistoryMessage]) -> str:
        """Combine the chat history into a single string
        with the LLM role tokens applied."""

        history = ""
        # Add the history messages to the history string
        for m in history:
            if m.role == ChatRole.HUMAN:
                role_prefix = ModelsRoleTokens.intel_neural_chat.human
            elif m.role == ChatRole.ASSISTANT:
                role_prefix = ModelsRoleTokens.intel_neural_chat.assistant
            else:
                raise ValueError(f"Invalid role: {m.role}")

            history += role_prefix + ": " + m.content + "\n"
        
        return history

    def append_to_history(self, content: str, role: ChatRole):
        """Append a new message to the history."""
        self.history.append(HistoryMessage(content=content, role=role))
    
    def generate_output_with_history(self, message: str) -> str:
        """Generate the output for the user message considering the history"""
        # Append the user's message to the history
        
        logger.info(f"Received message: {message}")
        self.append_to_history(content=message, role=ChatRole.HUMAN)
        
        # Use history to generate a response
        history_str = self.generate_chat_history_string(self.history)
        
        print(f"History: {history_str}")
        tool_decision_output = self.decide_rules_tool(self.lm, history_str)
        
        # Generate tool output and append it to the history
        output = self.generate_tool_output(tool_decision_output)
        self.append_to_history(content=output, role=ChatRole.ASSISTANT)
        
        return output