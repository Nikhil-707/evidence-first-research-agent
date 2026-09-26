"""Compare pipeline and baseline evaluation results.

Reads the output files from ``eval.run_eval`` and ``eval.run_baseline`` and
writes a Markdown summary plus a chart when Matplotlib is available.

Usage:
        python -m eval.compare_results
"""
import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent


def main():
    pipeline = json.loads((EVAL_DIR / "results_pipeline.json").read_text())
    baseline = json.loads((EVAL_DIR / "results_baseline.json").read_text())

    baseline_by_id = {b["id"]: b for b in baseline}

    rows = []
    for p in pipeline:
        b = baseline_by_id.get(p["id"])
        if not p["scoreable"] or not b or "answer_correctness" not in b:
            continue
        rows.append((p["id"], p.get("answer_correctness", 0), b["answer_correctness"], p.get("faithfulness", 0)))

    if not rows:
        print("No overlapping scoreable results found — run both eval scripts first.")
        return

    avg_pipeline = sum(r[1] for r in rows) / len(rows)
    avg_baseline = sum(r[2] for r in rows) / len(rows)
    avg_faith = sum(r[3] for r in rows) / len(rows)

    md = ["# Multi-Agent Pipeline vs. Single-Shot Baseline\n",
          f"Averaged over {len(rows)} scoreable benchmark questions.\n",
          "| Metric | Multi-Agent Pipeline | Single-Shot Baseline |",
          "|---|---|---|",
          f"| Answer correctness | **{avg_pipeline:.2f}** | {avg_baseline:.2f} |",
          f"| Faithfulness (grounding to sources) | **{avg_faith:.2f}** | N/A — no retrieval, nothing to ground against |",
          "\n## Per-question breakdown\n",
          "| ID | Pipeline correctness | Baseline correctness | Pipeline faithfulness |",
          "|---|---|---|---|"]
    for qid, pc, bc, pf in rows:
        md.append(f"| {qid} | {pc:.2f} | {bc:.2f} | {pf:.2f} |")

    (EVAL_DIR / "comparison.md").write_text("\n".join(md))
    print("\n".join(md))

    try:
        import matplotlib.pyplot as plt
        ids = [r[0] for r in rows]
        x = range(len(ids))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - width / 2 for i in x], [r[1] for r in rows], width, label="Multi-Agent Pipeline")
        ax.bar([i + width / 2 for i in x], [r[2] for r in rows], width, label="Single-Shot Baseline")
        ax.set_ylabel("Answer correctness (RAGAS)")
        ax.set_title("Multi-Agent Pipeline vs. Single-Shot Baseline")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ids, rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(EVAL_DIR / "comparison.png", dpi=150)
        print(f"\nChart saved to {EVAL_DIR / 'comparison.png'}")
    except ImportError:
        print("\n(matplotlib not installed — skipping chart; markdown table is still written.)")


if __name__ == "__main__":
    main()
