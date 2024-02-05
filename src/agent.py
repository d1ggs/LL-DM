from typing import Optional
import os

from guidance import gen, models, select
from llama_index.llms import OpenAILike

from src.vector_store import AutoMergingSRDIndex, SRDConfig
from loguru import logger
from enum import Enum
from guidance.models import LlamaCppChat
from llama_cpp import Llama
from llama_index.llms import LlamaCPP


class Model(Enum):
    LLAMA_CPP = 0
    OPENAI_API = 1


class GuidanceDMAgent:
    """DM Agent based on Guidance library"""
    
    def __init__(self, model_path: str, srd_config: Optional[SRDConfig] = None):
        self.model_path = model_path
        logger.debug("Loading SRD index")
        self.llama_index_model = self.load_llama_model()
        self.srd = self.load_srd_index(srd_config if srd_config is not None else SRDConfig())
        self.lm = LlamaCppChat(self.llama_index_model._model)

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
            context_window=1024*32,
            model_kwargs={"n_gpu_layers": -1}
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
            f"### System: You are a dungeon master for D&D 5e games. You have the following tools available: ('rules-database', Answers questions about the rules and details of the game). Do not rely on your own knowledge, use the tools to answer the question.\n"
            f"### User: {message}\n"
            f"### Assistant: I {select(['need', 'do not need'], name='choice')} to use a tool to answer your question."
        )

    def generate_tool_output(self, tool_decision_output):
        """Based on the decision on whether to use a tool, 
        generate the output accordingly."""
        
        if tool_decision_output["choice"] == "need":
            # Generate the question to be fed to the tool
            question_query = tool_decision_output + f"\n### Assistant: I will use the 'rules-database' tool to answer your question. The reworded question, wrapped in <question></question> to be fed to the tool is <question>{gen(name='query', stop='</question>')}'."
            # Use the tool to answer the question
            tool_answer = self.srd.query(question_query["query"]).response
                    
            answer = tool_decision_output +  f"### Assistant: I will use the 'rules-database' tool to answer your question. The tool's answer is {tool_answer}. \n\nMy final answer is then: {gen(stop='### User:', name='answer')}"
        else:
            answer = tool_decision_output + f'### Assistant: I do not need to use a tool to answer your question. {gen(stop="### User:", name="answer")}'
            
        return answer["answer"]
    
    def generate_output_no_history(self, message):
        """Generate the output for the user message with no history"""
        tool_decision_output = self.decide_rules_tool(self.lm, message)
        return self.generate_tool_output(tool_decision_output)
