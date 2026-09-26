from typing import TypedDict, List, Dict, Any, Optional

class SourceDoc(TypedDict):
    source_id: str
    sub_query: str
    title: str
    url: str
    snippet: str

class AgentState(TypedDict):
    query: str
    plan: List[str]
    raw_research: List[SourceDoc]
    critic_verdict: str          # "APPROVED" or "RETRY"
    critic_feedback: str          # Human-readable reasoning
    missing_facts: List[str]      # Facts requiring follow-up search
    retry_count: int
    final_report: str
    citations_used: List[str]     # Source IDs cited in report
    active_agent: str
    trace_log: List[Dict[str, Any]]
    llm_config: Optional[Dict[str, str]]  # ONLY stores {"provider": "...", "model": "..."} — NO KEYS!