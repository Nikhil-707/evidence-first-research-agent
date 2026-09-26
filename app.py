import os
import asyncio
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Evidence-First Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white; border: none; padding: 12px 24px;
        font-weight: 600; border-radius: 8px; transition: all 0.3s ease;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ System Status")

    provider = os.getenv("LLM_PROVIDER", "ollama").upper()
    model_name = os.getenv("GROQ_MODEL" if provider == "GROQ" else "OLLAMA_MODEL", "qwen2.5:1.5b")

    st.info(f"**LLM Backend:** `{provider}`")
    st.info(f"**Active Model:** `{model_name}`")

    st.divider()
    st.markdown("### 🎯 Pipeline Architecture")
    # Keep these descriptions aligned with the implemented workflow stages.
    st.caption("1. **Planner:** LLM-based query decomposition into sub-questions")
    st.caption("2. **Researcher:** Structured web search, URL-attributed sources")
    st.caption("3. **Critic:** Per-claim verification against cited sources, loops back on gaps")
    st.caption("4. **Writer:** Synthesizes answer with inline source citations")

st.title("⚡ Evidence-First Research Agent")
st.caption("Multi-agent research pipeline with source-grounded citations and a self-correcting verification loop — built with LangGraph.")

query = st.text_area(
    "Query Input:",
    height=100,
    value="When did India's Education Minister most recently resign, and why?"
)

if st.button("🚀 Run Research Pipeline"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        status_box = st.status("🔄 Initializing pipeline...", expanded=True)
        try:
            from src.agents.graph import graph_app

            initial_state = {
                "query": query, "plan": [], "raw_research": [],
                "critic_verdict": "", "critic_feedback": "", "missing_facts": [],
                "retry_count": 0, "final_report": "", "citations_used": [],
                "active_agent": "Start", "trace_log": [],
            }

            status_box.write("🧠 **Planner:** Decomposing query...")
            # Streamlit runs this handler synchronously; the graph itself is async.
            final_state = asyncio.run(graph_app.ainvoke(initial_state))
            status_box.update(label="✅ Pipeline complete", state="complete", expanded=False)

            st.divider()
            tab1, tab2, tab3 = st.tabs(["📄 Final Report", "🛠️ Agent Trace", "🔍 Sources"])

            with tab1:
                st.markdown("### Synthesized, Cited Answer")
                st.success(final_state.get("final_report", "No response generated."))

                cited_ids = final_state.get("citations_used", [])
                sources = {s["source_id"]: s for s in final_state.get("raw_research", [])}
                if cited_ids:
                    st.markdown("#### Sources cited above")
                    for cid in cited_ids:
                        s = sources.get(cid)
                        if s and s["url"]:
                            st.markdown(f"**[{cid}]** [{s['title']}]({s['url']})")

            with tab2:
                st.markdown("### Agent Execution Trace")
                trace = final_state.get("trace_log", [])
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Steps", len(trace))
                col2.metric("Critic Verdict", final_state.get("critic_verdict", "N/A"))
                col3.metric("Retry Loop Count", final_state.get("retry_count", 0))
                st.divider()
                for step in trace:
                    agent = step.get("agent", "Agent")
                    with st.expander(f"📌 Step: **{agent}**", expanded=True):
                        st.json(step)

            with tab3:
                st.markdown("### All Retrieved Sources")
                for s in final_state.get("raw_research", []):
                    with st.expander(f"[{s['source_id']}] {s['title']}"):
                        st.write(s["url"])
                        st.write(s["snippet"])

        except Exception as e:
            status_box.update(label="❌ Execution Failed", state="error")
            st.error(f"Error: {str(e)}")
