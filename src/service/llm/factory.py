from src.service.llm.openai_llm import OpenAIChatLLM

def get_llm_from_request(provider: str, model: str, api_key: str):
    if provider == "openai":
        return OpenAIChatLLM(model=model, 
                              api_key=api_key
        )
    
    # elif provider == "anthropic":
    #     return AnthropicLLM(model=model, api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")