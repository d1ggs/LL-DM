from typing import Optional

from guidance import gen, models, select
from llama_index.llms import OpenAILike

from src.vector_store import AutoMergingSRDIndex, SRDConfig


class GuidanceDMAgent:
    def __init__(self, srd_config: Optional[SRDConfig] = None):
        self.srd = self.load_srd_index(srd_config if srd_config is not None else SRDConfig())
        self.lm = models.LlamaCppChat(model="/mnt/d/WSL/llamacpp/models/neural-chat-7b-v3-1.Q5_K_M.gguf",
                                      n_gpu_layers=-1,
                                      n_ctx=1024*32)
        
    def load_srd_index(self, srd_config: SRDConfig):
        """Loads the Standard Reference Document (SRD) index"""
        fake_openai = OpenAILike(api_base="http://localhost:5000/v1", api_key="...")
        index = AutoMergingSRDIndex(fake_openai, srd_config)
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
            answer = tool_decision_output + f'### Assistant: {gen(stop="### User:", name="answer")}'
            
        return answer["answer"]
    
    def generate_output_no_history(self, message):
        """Generate the output for the user message with no history"""
        tool_decision_output = self.decide_rules_tool(self.lm, message)
        return self.generate_tool_output(tool_decision_output)
