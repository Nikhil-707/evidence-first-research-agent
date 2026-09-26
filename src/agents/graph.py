import json
from typing import Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.config import get_llm
from src.tools.mcp_client import get_mcp_search_tool
from src.agents.state import AgentState, SourceDoc

llm = get_llm()


# --- PYDANTIC SCHEMAS FOR DETERMINISTIC PARSING ---
class PlanSchema(BaseModel):
    sub_queries: List[str] = Field(description="2-3 specific, targeted search queries that together would answer the user's question.")


class CriticSchema(BaseModel):
    verdict: Literal["APPROVED", "RETRY"] = Field(
        description="APPROVED only if every part of the user's question is directly supported by a specific cited source. RETRY if any part is unsupported, ambiguous, or missing."
    )
    missing_facts: List[str] = Field(
        default_factory=list,
        description="Specific facts/entities still missing or unverified. Empty if APPROVED. Used to generate a better follow-up search, so be concrete (e.g. 'the exact date of resignation', not 'more detail')."
    )
    reasoning: str = Field(description="One sentence on why this verdict was reached, referencing source_ids where possible.")


class ReformulateSchema(BaseModel):
    new_queries: List[str] = Field(description="1-2 new, more targeted search queries designed specifically to find the missing facts.")


def _parse_mcp_result(raw) -> List[dict]:
    """MCP tools return a list of content blocks (e.g. {'type': 'text',
    'text': '<json string>'}) rather than plain Python objects — that's the
    protocol boundary. Unwrap it back into the list[dict] our nodes expect."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        try:
            return json.loads(raw[0]["text"])
        except (json.JSONDecodeError, IndexError, KeyError):
            return [{"title": "Parse error", "url": "", "snippet": str(raw)}]
    if isinstance(raw, list):
        return raw
    return [{"title": "Unexpected result", "url": "", "snippet": str(raw)}]


def _format_sources(sources: List[SourceDoc]) -> str:
    """Renders sources with explicit IDs so the Critic/Writer can attribute
    claims to a specific source instead of treating research as one blob."""
    if not sources:
        return "No sources retrieved yet."
    lines = []
    for s in sources:
        lines.append(f"[{s['source_id']}] ({s['url']}) {s['title']}\n{s['snippet']}")
    return "\n\n".join(lines)


# --- NODE 1: PLANNER ---
async def planner_node(state: AgentState) -> dict:
    query = state["query"]
    prompt = f"""Break this question into 2-3 concise, targeted search queries that would let you find every fact needed to answer it fully.
Extract key entities (names, dates, events) rather than repeating the whole question verbatim.

User Query: {query}"""

    try:
        structured_llm = llm.with_structured_output(PlanSchema)
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        plan = res.sub_queries if res and res.sub_queries else [query]
    except Exception:
        # Fall back to the original query if structured planning fails.
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

    if retry_count > 0 and missing_facts:
        prompt = f"""The following facts are still missing to fully answer the user's question.
Generate 1-2 new, specific search queries designed to find exactly these missing facts.

User Query: {state['query']}
Missing facts: {missing_facts}"""
        try:
            structured_llm = llm.with_structured_output(ReformulateSchema)
            res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            queries_to_run = res.new_queries if res and res.new_queries else missing_facts
        except Exception:
            queries_to_run = missing_facts
    else:
        queries_to_run = state.get("plan", [state["query"]])

    # Load the MCP search tool once and reuse it across planned queries.
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

    all_sources = existing_sources + new_sources  # accumulate rather than discard on retry

    log = state.get("trace_log", [])
    log.append({"agent": "Researcher", "action": "Ran web searches", "queries": queries_to_run, "new_sources": len(new_sources)})

    return {"raw_research": all_sources, "active_agent": "Researcher", "trace_log": log}


# --- NODE 3: CRITIC ---
async def critic_node(state: AgentState) -> dict:
    query = state["query"]
    sources = state.get("raw_research", [])
    retry_count = state.get("retry_count", 0)

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

    prompt = f"""You are verifying whether the research below is sufficient to fully and specifically answer the user's question.
Check each part of the question against the cited sources individually — do not approve based on a general impression.
If a specific date, name, or number is asked for and no source states it explicitly, that is grounds for RETRY.

User Query: {query}

Sources:
{_format_sources(sources)}"""

    try:
        structured_llm = llm.with_structured_output(CriticSchema)
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        verdict = res.verdict
        missing_facts = res.missing_facts
        reasoning = res.reasoning
    except Exception:
        verdict = "APPROVED"
        missing_facts = []
        reasoning = "Critic call failed; defaulting to APPROVED to avoid an infinite loop."

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

    prompt = f"""Answer the user's question using ONLY the sources below. Every factual claim must end with
its source id in brackets, e.g. "...resigned on July 25, 2026 [S2]." If sources conflict, say so explicitly.
Do not state anything not supported by a source.

User Query: {query}

Sources:
{_format_sources(sources)}"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    report = response.content

    # Extract which source_ids actually got cited, so the UI can render a
    # real "Sources used" section instead of dumping every retrieved doc.
    cited = sorted({s["source_id"] for s in sources if f"[{s['source_id']}]" in report}, key=lambda x: int(x[1:]))

    log = state.get("trace_log", [])
    log.append({"agent": "Writer", "action": "Synthesized cited answer", "citations_used": cited})

    return {"final_report": report, "citations_used": cited, "active_agent": "Writer", "trace_log": log}


def route_critic(state: AgentState) -> Literal["researcher", "writer"]:
    if state.get("critic_verdict") == "RETRY":
        return "researcher"
    return "writer"


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
