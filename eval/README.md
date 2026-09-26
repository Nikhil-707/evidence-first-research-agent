# Evaluation Guide

This benchmark compares the research workflow with a single-call model baseline. It measures RAGAS faithfulness for the agent's retrieved contexts and answer correctness for questions with reference answers. The evaluation scripts manage their asynchronous workflow internally.

## Run the evaluation

Configure Ollama or Groq in `.env`, confirm that the selected model is available, and run these commands from the repository root:

```bash
python -m eval.run_eval
python -m eval.run_baseline
python -m eval.compare_results
```

The scripts write `eval/results_pipeline.json` and `eval/results_baseline.json`. The comparison command uses those files to create `eval/comparison.md` and, when Matplotlib is available, `eval/comparison.png`.

These runs call the configured language model, perform public web searches, and download/use a local Hugging Face embedding model. They are not guaranteed to be free: provider pricing and usage limits vary. Search results and model outputs may also change between runs.

## Benchmark coverage

`benchmark.json` contains questions about recent events, factual questions with reference answers, and open-ended prompts whose reference text begins with `OPEN —` or `Check latest`. Open-ended questions are not scored for answer correctness. Review their outputs manually, especially items `q4`, `q8`, and `q15`, to see whether the system communicates uncertainty appropriately.

## Interpreting results

The benchmark is a development aid, not a general certification of accuracy or a controlled scientific study. Scores describe this dataset, selected provider/model, and run conditions. They should not be used to claim that the agent consistently outperforms a standalone model on all tasks.

Reference answers for time-sensitive questions can become outdated. Review and update those entries before relying on comparisons, and note the date and model/provider when sharing results. The evaluation intentionally supports both workflow and baseline runs so changes can be compared under similar conditions.
