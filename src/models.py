
from enum import Enum

from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI


class ModelType(Enum):
    LLAMA_CPP = "llama_cpp"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    LOCAL_API = "text-generation-webui"
    

class ModelFactory:
    llama_cpp_default_kwargs = {
        "model_path": "models/neural-chat-7b-v3-3.Q5_K_M.gguf",
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.95,
        "verbose": True, # Verbose is required to pass to the callback manager
        "n_gpu_layers": -1,
        "echo": True,
        "n_ctx": 1024*32,
        "stop": ["### System", "### User", "### Assistant", "User:", "Player:", "DM:"],
    }
    
    gpt_default_kwargs = {
        "temperature": 0.1,
        "max_tokens": 512,
        "streaming": True,
    }
    
    local_api_default_kwargs = {
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.95,
        "n_gpu_layers": -1,
        "echo": True,
        "stop": ["### System", "### User", "### Assistant", "User:", "Player:", "DM:"],
        "streaming": True,
    }
    
    @classmethod
    def get_model(cls, model_type: ModelType, **kwargs):
        """Get a model of the specified type with the specified kwargs"""
        
        # Local model running through LLaMaCpp
        if model_type == ModelType.LLAMA_CPP:
            args = cls.llama_cpp_default_kwargs.copy()
            args.update(kwargs)
            model = LlamaCpp(**args)
        
        elif model_type == ModelType.GPT_35_TURBO:
            args = cls.gpt_default_kwargs.copy()
            args.update(kwargs)
            model = ChatOpenAI(model="gpt-3.5-turbo-0613", **args)
            
        elif model_type == ModelType.GPT_4:
            args = cls.gpt_default_kwargs.copy()
            args.update(kwargs)
            model = ChatOpenAI(model="gpt-4-1106-preview", **kwargs)
        
        elif model_type == ModelType.LOCAL_API:
            args = cls.local_api_default_kwargs.copy()
            args.update(kwargs)
            model = ChatOpenAI(model="text-generation-webui", 
                               openai_api_base="http://localhost:5000/v1/", 
                               **kwargs)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
            
        return model