"""Run the benchmark with single-call model responses as a baseline.

This baseline does not plan searches, retrieve web results, or produce
retrieval-based citations. Answer correctness is scored only where a
concrete reference answer is available.

Usage:
    python -m eval.run_baseline
Output:
    eval/results_baseline.json
"""
import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_core.messages import HumanMessage

from src.config import get_llm, get_embeddings

BENCH_PATH = Path(__file__).parent / "benchmark.json"
OUT_PATH = Path(__file__).parent / "results_baseline.json"
OPEN_ENDED_MARKERS = ("OPEN —", "Check latest")


def main():
    benchmark = json.loads(BENCH_PATH.read_text())
    llm = get_llm()

    records = []
    for item in benchmark:
        print(f"  -> {item['id']}: {item['question'][:70]}...")
        response = llm.invoke([HumanMessage(content=item["question"])])
        records.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": response.content,
            "scoreable": not item["ground_truth"].startswith(OPEN_ENDED_MARKERS),
        })

    judge = LangchainLLMWrapper(llm)
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    scoreable = [r for r in records if r["scoreable"]]
    if scoreable:
        ds = Dataset.from_list([{
            "question": r["question"], "answer": r["answer"], "ground_truth": r["ground_truth"],
        } for r in scoreable])
        scores = evaluate(ds, metrics=[answer_correctness], llm=judge, embeddings=embeddings)
        for i, r in enumerate(scoreable):
            r["answer_correctness"] = scores["answer_correctness"][i]

    OUT_PATH.write_text(json.dumps(records, indent=2))

    if scoreable:
        avg = sum(r["answer_correctness"] for r in scoreable) / len(scoreable)
        print(f"\nBaseline average answer_correctness: {avg:.2f}")
    print(f"Results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
