"""Run the research workflow benchmark and score results with RAGAS.

Faithfulness is measured for all records. Answer correctness is measured
only for records with concrete reference answers. Open-ended records are
retained for qualitative review.

Usage:
    python -m eval.run_eval
Output:
    eval/results_pipeline.json
"""
import json
import time
from pathlib import Path

from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from src.config import get_llm, get_embeddings
from src.agents.graph import graph_app

BENCH_PATH = Path(__file__).parent / "benchmark.json"
OUT_PATH = Path(__file__).parent / "results_pipeline.json"

OPEN_ENDED_MARKERS = ("OPEN —", "Check latest")


async def run_pipeline(query: str) -> dict:
    initial_state = {
        "query": query, "plan": [], "raw_research": [],
        "critic_verdict": "", "critic_feedback": "", "missing_facts": [],
        "retry_count": 0, "final_report": "", "citations_used": [],
        "active_agent": "Start", "trace_log": [],
    }
    result = await graph_app.ainvoke(initial_state)
    contexts = [f"[{s['source_id']}] {s['snippet']}" for s in result.get("raw_research", [])]
    return {
        "answer": result.get("final_report", ""),
        "contexts": contexts if contexts else ["No context retrieved."],
        "retry_count": result.get("retry_count", 0),
    }


async def _main():
    benchmark = json.loads(BENCH_PATH.read_text())

    scoreable, open_ended = [], []
    for item in benchmark:
        (open_ended if item["ground_truth"].startswith(OPEN_ENDED_MARKERS) else scoreable).append(item)

    print(f"Running pipeline on {len(benchmark)} questions "
          f"({len(scoreable)} scoreable with ground truth, {len(open_ended)} open-ended)...\n")

    records = []
    for item in benchmark:
        print(f"  -> {item['id']}: {item['question'][:70]}...")
        t0 = time.time()
        out = await run_pipeline(item["question"])
        elapsed = time.time() - t0
        records.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": out["answer"],
            "contexts": out["contexts"],
            "retry_count": out["retry_count"],
            "latency_sec": round(elapsed, 1),
            "scoreable": item in scoreable,
        })

    # Use the configured chat model as judge and local embeddings for scoring.
    judge = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    ds = Dataset.from_list([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    } for r in records])

    print("\nScoring with RAGAS (faithfulness on all, answer_correctness on scoreable subset)...")
    all_scores = evaluate(ds, metrics=[faithfulness], llm=judge, embeddings=embeddings)

    scoreable_idx = [i for i, r in enumerate(records) if r["scoreable"]]
    correctness_scores = None
    if scoreable_idx:
        ds_scoreable = ds.select(scoreable_idx)
        correctness_scores = evaluate(ds_scoreable, metrics=[answer_correctness], llm=judge, embeddings=embeddings)

    for i, r in enumerate(records):
        r["faithfulness"] = all_scores["faithfulness"][i]
    if correctness_scores:
        for j, i in enumerate(scoreable_idx):
            records[i]["answer_correctness"] = correctness_scores["answer_correctness"][j]

    OUT_PATH.write_text(json.dumps(records, indent=2))

    avg_faith = sum(r["faithfulness"] for r in records) / len(records)
    print(f"\nAverage faithfulness: {avg_faith:.2f}")
    if correctness_scores:
        avg_correct = sum(records[i].get("answer_correctness", 0) for i in scoreable_idx) / len(scoreable_idx)
        print(f"Average answer_correctness (scoreable subset): {avg_correct:.2f}")
    print(f"\nFull results written to {OUT_PATH}")


def main():
    import asyncio
    asyncio.run(_main())


if __name__ == "__main__":
    main()
