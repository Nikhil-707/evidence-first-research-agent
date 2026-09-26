from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.graph import graph_app

app = FastAPI(title="DevAgent Research API")


def _initial_state(query: str) -> dict:
    return {
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
    }


class ResearchRequest(BaseModel):
    query: str


@app.post("/api/research")
async def run_research(req: ResearchRequest):
    try:
        result = await graph_app.ainvoke(_initial_state(req.query))
        return {
            "query": req.query,
            "report": result.get("final_report"),
            "citations_used": result.get("citations_used"),
            "sources": result.get("raw_research"),
            "trace_log": result.get("trace_log"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
