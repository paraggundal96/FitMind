import os
from langchain_core.language_models import model_profile
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

models = [
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-12b-it:free",
    "qwen/qwen3-coder:free"
]

def get_openrouter_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )

def get_ollama_llm():
    return ChatOllama(
        model = "phi3:mini",
        model_provider = "ollama",
    
    )

def invoke_(prompt):
    last_error = None

    # ---- Try OpenRouter models ----
    for model in models:
        try:
            print(f"Trying model: {model}")
            llm = get_openrouter_llm(model)
            response = llm.invoke(prompt)
            print(f"Success with: {model}")
            return response

        except Exception as e:
            print(f"Failed with {model}: {str(e)}")
            last_error = e

    # ---- अंतिम fallback: Ollama ----
    try:
        print("Falling back to local Ollama (phi3:mini)")
        llm = get_ollama_llm()
        response = llm.invoke(prompt)
        return response

    except Exception as e:
        print("Ollama also failed")
        raise RuntimeError("All models failed") from e


# ---- Usage ----
response = invoke_("Explain Attention Mechanism in short") 
print(response.content)