import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_llm() -> BaseChatModel:
    """
    Returns the active chat model based on LLM_PROVIDER env var.
    'ollama' -> local dev (default, zero cost, private)
    'groq'   -> free-tier cloud inference for the deployed public demo
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com (no card required)."
            )
        # NOTE: Groq's free-tier catalog does not include Qwen models.
        # llama-3.3-70b-versatile is confirmed available on the free,
        # no-card developer tier as of this writing.
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.1)

    else:
        # Default: Local Ollama
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1,
            validate_model_on_init=False,  # so imports don't block if Ollama isn't running yet
        )


def get_embeddings():
    """
    Free, local embedding model (no API key, no cost) used by the eval
    harness for context-precision-style metrics. Runs on CPU via
    sentence-transformers.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
