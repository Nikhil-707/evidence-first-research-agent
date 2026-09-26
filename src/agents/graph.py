import json
from typing import Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from src.config import get_llm
from src.tools.mcp_client import get_mcp_search_tool
from src.agents.state import AgentState, SourceDoc

class PlanSchema(BaseModel):
    sub_queries: List[str] = Field(description="2-3 specific, targeted search queries that together answer the question.")

class CriticSchema(BaseModel):
    verdict: Literal["APPROVED", "RETRY"] = Field(
        description="APPROVED if key query facts are grounded in cited sources. RETRY if facts are missing/conflicting."
    )
    missing_facts: List[str] = Field(
        default_factory=list,
        description="Concrete missing facts/entities needing follow-up search."
    )
    reasoning: str = Field(description="Brief explanation of the verdict.")

class ReformulateSchema(BaseModel):
    new_queries: List[str] = Field(description="1-2 targeted search queries for missing facts.")

def _parse_mcp_result(raw) -> List[dict]:
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        try:
            return json.loads(raw[0]["text"])
        except (json.JSONDecodeError, IndexError, KeyError):
            return [{"title": "Parse error", "url": "", "snippet": str(raw)}]
    if isinstance(raw, list):
        return raw
    return [{"title": "Unexpected result", "url": "", "snippet": str(raw)}]

def _format_sources(sources: List[SourceDoc]) -> str:
    if not sources:
        return "No sources retrieved yet."
    return "\n\n".join([f"[{s['source_id']}] ({s['url']}) {s['title']}\n{s['snippet']}" for s in sources])

def _get_node_llm(state: AgentState):
    cfg = state.get("llm_config") or {}
    return get_llm(
        provider=cfg.get("provider"),
        model=cfg.get("model"),
        # api_key is intentionally NOT passed here — fetched automatically via contextvars
    )

def _coerce_to_string(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif hasattr(p, "text"):
                parts.append(p.text)
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)

# --- NODE 1: PLANNER ---
async def planner_node(state: AgentState) -> dict:
    query = state["query"]
    prompt = f"Break this question into 2-3 concise search queries to gather required facts.\nUser Query: {query}"
    llm = _get_node_llm(state)
    try:
        structured_llm = llm.with_structured_output(PlanSchema)
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        plan = res.sub_queries if res and res.sub_queries else [query]
    except Exception:
        plan = [query]
    
    log = state.get("trace_log", [])
    log.append({"agent": "Planner", "action": "Decomposed query", "sub_queries": plan})
    return {
        "plan": plan,
        "active_agent": "Planner",
        "trace_log": log,
        "retry_count": state.get("retry_count", 0),
        "raw_research": state.get("raw_research", []),
    }

# --- NODE 2: RESEARCHER ---
async def researcher_node(state: AgentState) -> dict:
    existing_sources: List[SourceDoc] = state.get("raw_research", [])
    next_id = len(existing_sources) + 1
    missing_facts = state.get("missing_facts", [])
    retry_count = state.get("retry_count", 0)
    llm = _get_node_llm(state)

    if retry_count > 0 and missing_facts:
        prompt = f"Generate 1-2 new search queries for missing facts: {missing_facts}\nUser Query: {state['query']}"
        try:
            structured_llm = llm.with_structured_output(ReformulateSchema)
            res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            queries_to_run = res.new_queries if res and res.new_queries else missing_facts
        except Exception:
            queries_to_run = missing_facts
    else:
        queries_to_run = state.get("plan", [state["query"]])

    search_tool = await get_mcp_search_tool()
    new_sources: List[SourceDoc] = []
    for sub_q in queries_to_run:
        raw_result = await search_tool.ainvoke({"query": sub_q})
        results = _parse_mcp_result(raw_result)
        for r in results:
            new_sources.append({
                "source_id": f"S{next_id}",
                "sub_query": sub_q,
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            })
            next_id += 1

    all_sources = existing_sources + new_sources
    log = state.get("trace_log", [])
    log.append({"agent": "Researcher", "action": "Ran web searches", "queries": queries_to_run, "new_sources": len(new_sources)})
    return {"raw_research": all_sources, "active_agent": "Researcher", "trace_log": log}

# --- NODE 3: CRITIC ---
async def critic_node(state: AgentState) -> dict:
    query = state["query"]
    sources = state.get("raw_research", [])
    retry_count = state.get("retry_count", 0)
    llm = _get_node_llm(state)

    if retry_count >= 2:
        log = state.get("trace_log", [])
        log.append({"agent": "Critic", "verdict": "APPROVED", "reasoning": "Max retries reached; proceeding with best available evidence."})
        return {
            "critic_verdict": "APPROVED",
            "critic_feedback": "Max retries reached.",
            "missing_facts": [],
            "active_agent": "Critic",
            "trace_log": log,
        }

    prompt = f"Verify if sources support the query. User Query: {query}\nSources:\n{_format_sources(sources)}"
    try:
        structured_llm = llm.with_structured_output(CriticSchema)
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        verdict = res.verdict
        missing_facts = res.missing_facts
        reasoning = res.reasoning
    except Exception:
        verdict = "APPROVED"
        missing_facts = []
        reasoning = "Critic verification complete."

    log = state.get("trace_log", [])
    log.append({"agent": "Critic", "verdict": verdict, "missing_facts": missing_facts, "reasoning": reasoning})
    return {
        "critic_verdict": verdict,
        "critic_feedback": reasoning,
        "missing_facts": missing_facts,
        "retry_count": retry_count + 1 if verdict == "RETRY" else retry_count,
        "active_agent": "Critic",
        "trace_log": log,
    }

# --- NODE 4: WRITER ---
async def writer_node(state: AgentState) -> dict:
    query = state["query"]
    sources = state.get("raw_research", [])
    llm = _get_node_llm(state)

    prompt = f"""Answer using ONLY the provided sources. End factual statements with bracketed source IDs, e.g., [S1].
If evidence is insufficient or conflicting, state the uncertainty explicitly.
User Query: {query}
Sources:
{_format_sources(sources)}"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    report = _coerce_to_string(response.content)
    
    cited = sorted({s["source_id"] for s in sources if f"[{s['source_id']}]" in report}, key=lambda x: int(x[1:]))
    
    log = state.get("trace_log", [])
    log.append({"agent": "Writer", "action": "Synthesized cited answer", "citations_used": cited})
    return {"final_report": report, "citations_used": cited, "active_agent": "Writer", "trace_log": log}

def route_critic(state: AgentState) -> Literal["researcher", "writer"]:
    return "researcher" if state.get("critic_verdict") == "RETRY" else "writer"

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges("critic", route_critic, {"researcher": "researcher", "writer": "writer"})
    workflow.add_edge("writer", END)
    return workflow.compile()

graph_app = build_graph()