from typing import TypedDict, List, Dict, Any


class SourceDoc(TypedDict):
    """A single retrieved document, kept structured (not flattened to text)
    so the Critic and Writer can attribute claims to a specific source_id
    and the UI can render real, clickable citations."""
    source_id: str      # e.g. "S1", "S2" — stable ID used in citations
    sub_query: str       # which planner sub-query produced this
    title: str
    url: str
    snippet: str


class AgentState(TypedDict):
    query: str
    plan: List[str]
    raw_research: List[SourceDoc]
    critic_verdict: str          # "APPROVED" or "RETRY"
    critic_feedback: str          # human-readable reason
    missing_facts: List[str]      # structured list of what's still unanswered (drives reformulation)
    retry_count: int
    final_report: str
    citations_used: List[str]     # source_ids the Writer actually cited
    active_agent: str
    trace_log: List[Dict[str, Any]]
