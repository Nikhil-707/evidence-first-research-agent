import os
import asyncio
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="DevAgent | Multi-Provider Fleet",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROVIDER_MODELS = {
    "Groq (Free Cloud)": "openai/gpt-oss-120b",
    "xAI Grok (console.x.ai)": "grok-2-1212",
    "Google Gemini": "gemini-2.5-flash",
    "OpenAI": "gpt-4o-mini",
    "Anthropic Claude": "claude-3-5-sonnet-20241022",
    "Hugging Face": "meta-llama/Llama-3.2-3B-Instruct",
    "Ollama (Free Local)": "qwen2.5:1.5b",
}

PROVIDER_KEYS = {
    "Groq (Free Cloud)": "groq",
    "xAI Grok (console.x.ai)": "xai",
    "Google Gemini": "gemini",
    "OpenAI": "openai",
    "Anthropic Claude": "anthropic",
    "Hugging Face": "huggingface",
    "Ollama (Free Local)": "ollama",
}

with st.sidebar:
    st.title("⚙️ LLM Engine Config")
    
    selected_provider_label = st.selectbox(
        "Select Provider",
        options=list(PROVIDER_MODELS.keys()),
        index=0
    )
    
    provider_code = PROVIDER_KEYS[selected_provider_label]
    default_model = PROVIDER_MODELS[selected_provider_label]
    
    model_name = st.text_input("Model ID / Name", value=default_model)
    
    api_key = ""
    if provider_code != "ollama":
        api_key = st.text_input(
            f"{selected_provider_label} API Key",
            type="password",
            help="Your key stays exclusively in memory for this session and is never written to disk or logs."
        )
        st.caption("🔒 **Security Assurance:** Keys are held in volatile session memory during execution.")

    st.divider()
    st.markdown("### 🎯 Pipeline Stages")
    st.caption("1. **Planner:** Decomposes queries into focused search goals")
    st.caption("2. **Researcher:** Fetches structured web hits via MCP client")
    st.caption("3. **Critic:** Evaluates evidence completeness and triggers retries")
    st.caption("4. **Writer:** Synthesizes response with bracketed source tags")

st.title("⚡ DevAgent: Autonomous Research Fleet")
st.caption("Source-grounded research pipeline running on LangGraph & MCP.")

query = st.text_area(
    "Research Topic / Query:",
    height=100,
    value="When did India's Education Minister most recently resign, and why?"
)

if st.button("🚀 Run Research Pipeline"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    elif provider_code != "ollama" and not api_key and not os.getenv(f"{provider_code.upper()}_API_KEY"):
        st.error(f"Please enter your {selected_provider_label} API key in the sidebar.")
    else:
        status_box = st.status("🔄 Running multi-agent research workflow...", expanded=True)
        try:
            from src.agents.graph import graph_app
            from src.config import set_session_api_key

            # Set session API key in Python contextvars (completely invisible to LangGraph state)
            set_session_api_key(api_key if api_key else None)

            initial_state = {
                "query": query,
                "plan": [],
                "raw_research": [],
                "critic_verdict": "",
                "critic_feedback": "",
                "missing_facts": [],
                "retry_count": 0,
                "final_report": "",
                "citations_used": [],
                "active_agent": "Start",
                "trace_log": [],
                "llm_config": {
                    "provider": provider_code,
                    "model": model_name
                    # api_key is intentionally EXCLUDED from state
                }
            }

            status_box.write(f"🧠 **Active Engine:** `{selected_provider_label}` ({model_name})...")
            final_state = asyncio.run(graph_app.ainvoke(initial_state))
            status_box.update(label="✅ Pipeline Execution Complete", state="complete", expanded=False)

            st.divider()
            tab1, tab2, tab3 = st.tabs(["📄 Final Report", "🛠️ Agent Trace", "🔍 Sources Collected"])

            with tab1:
                st.markdown("### Synthesized Answer")
                st.markdown(final_state.get("final_report", "No report generated."))

                cited_ids = final_state.get("citations_used", [])
                sources = {s["source_id"]: s for s in final_state.get("raw_research", [])}
                if cited_ids:
                    st.markdown("---")
                    st.markdown("#### Cited References")
                    for cid in cited_ids:
                        s = sources.get(cid)
                        if s and s["url"]:
                            st.markdown(f"**[{cid}]** [{s['title']}]({s['url']})")

            with tab2:
                st.markdown("### Execution Trace")
                trace = final_state.get("trace_log", [])
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Steps", len(trace))
                col2.metric("Critic Verdict", final_state.get("critic_verdict", "N/A"))
                col3.metric("Retry Loops", final_state.get("retry_count", 0))
                st.divider()
                for step in trace:
                    agent = step.get("agent", "Agent")
                    with st.expander(f"📌 Step: **{agent}**", expanded=True):
                        st.json(step)

            with tab3:
                st.markdown("### All Retrieved Sources")
                for s in final_state.get("raw_research", []):
                    with st.expander(f"[{s['source_id']}] {s['title']}"):
                        st.write(f"**URL:** {s['url']}")
                        st.write(f"**Snippet:** {s['snippet']}")

        except Exception as e:
            status_box.update(label="❌ Execution Failed", state="error")
            err_str = str(e)

            if any(term in err_str.lower() for term in ["model_decommissioned", "model_not_found", "404", "invalid_request_error"]):
                st.error("⚠️ **Model Error / Key Mismatch**")
                st.info(
                    "The model ID was rejected by the provider. Please verify your provider selection and API key:\n\n"
                    "• **If your API key is from `console.x.ai` (starts with `xai-`):** Select **xAI Grok (console.x.ai)** and use Model ID `grok-2-1212`.\n"
                    "• **If your API key is from `console.groq.com` (starts with `gsk_`):** Select **Groq (Free Cloud)** and use Model ID `openai/gpt-oss-120b`."
                )
            else:
                st.error(f"Execution Error: {err_str}")