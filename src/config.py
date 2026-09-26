import os
import contextvars
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

# Task-isolated in-memory key storage (completely invisible to LangGraph state)
_SESSION_API_KEY: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_api_key", default=None)

def set_session_api_key(key: Optional[str]) -> None:
    """Sets the API key for the current execution thread/task context."""
    _SESSION_API_KEY.set(key)

def get_session_api_key() -> Optional[str]:
    """Retrieves the active session API key."""
    return _SESSION_API_KEY.get()

def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
    
    # Priority: Explicit argument -> ContextVar session key -> Environment variable
    resolved_key = api_key or get_session_api_key()

    if provider in ["xai", "grok"]:
        from langchain_openai import ChatOpenAI
        key = resolved_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not key:
            raise ValueError("xAI Grok API Key is required. Please enter it in the sidebar.")
        model_name = model or os.getenv("XAI_MODEL", "grok-2-1212")
        return ChatOpenAI(
            api_key=key,
            model=model_name,
            openai_api_base="https://api.x.ai/v1",
            temperature=0.1
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = resolved_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key is required. Please enter it in the sidebar.")
        model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(api_key=key, model_name=model_name, temperature=0.1)

    elif provider in ["gemini", "google"]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = resolved_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Google Gemini API Key is required. Please enter it in the sidebar.")
        model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(google_api_key=key, model_name=model_name, temperature=0.1)

    elif provider in ["openai", "chatgpt"]:
        from langchain_openai import ChatOpenAI
        key = resolved_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is required. Please enter it in the sidebar.")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(api_key=key, model=model_name, temperature=0.1)

    elif provider in ["anthropic", "claude"]:
        from langchain_anthropic import ChatAnthropic
        key = resolved_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API Key is required. Please enter it in the sidebar.")
        model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(api_key=key, model_name=model_name, temperature=0.1)

    elif provider in ["huggingface", "hf"]:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        key = resolved_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not key:
            raise ValueError("Hugging Face API Token is required. Please enter it in the sidebar.")
        repo_id = model or os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=key,
            temperature=0.1,
            task="text-generation"
        )
        return ChatHuggingFace(llm=endpoint)

    else:
        from langchain_ollama import ChatOllama
        model_name = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            base_url=url,
            temperature=0.1,
            validate_model_on_init=False,
        )

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")