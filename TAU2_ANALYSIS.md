# Analyze tau2 failures with Jacobian Lens

The `scripts/analyze_tau2.py` command reads both `results.json` and the verbose
agent-call logs saved below `artifacts/task_*/sim_*/llm_debug/`. By default it
analyzes failed simulations and the final `agent_response` call from each one.

It reconstructs the request with the Qwen chat template, teacher-forces each
unmet expected tool name and the tool name actually emitted, and writes their
layer-wise log-probabilities and ranks. It also produces an interactive slice
of the exact pre-response context.

## Vast.ai setup

Stop the vLLM server before the GPU analysis so Qwen can be loaded by PyTorch.
The existing 32 GB disk is too tight for another environment, the lens, and
reports; expand the instance disk to at least 50 GB first. Copy this modified
repository to `/workspace/jacobian-lens`, then run:

```bash
set -euo pipefail

export PATH="/root/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"

cd /workspace/jacobian-lens
uv sync --frozen

RUN_DIR="/workspace/tau2-bench/data/simulations/qwen3-8b-airline-10tasks-nonthinking-seed300"
OUT_DIR="/workspace/qwen3-8b-jlens-analysis"
MODEL_REVISION="$(tr -d '\r\n' < /workspace/qwen3-8b-revision.txt)"

uv run python scripts/analyze_tau2.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --model-revision "$MODEL_REVISION" \
  --inspect-only
```

`--inspect-only` does not load Qwen or the lens. Check that `selected_calls` is
non-zero and that the cases do not say `missing_agent_logs` in
`$OUT_DIR/manifest.json`.

First run one failed case without HTML as a GPU and tokenization check:

```bash
uv run python scripts/analyze_tau2.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR/pilot" \
  --model-revision "$MODEL_REVISION" \
  --limit 1 \
  --no-html \
  --layer-stride 4 \
  --max-seq-len 32768
```

Then analyze every saved case, including successes for comparison:

```bash
uv run python scripts/analyze_tau2.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR/full" \
  --model-revision "$MODEL_REVISION" \
  --include-successes \
  --call-selection last \
  --layer-stride 4 \
  --last-n-tokens 96 \
  --max-seq-len 32768
```

Do not add `--enable-thinking` for the run above: tau2 generated it with
`enable_thinking=false`.

Serve the interactive pages over HTTP rather than opening `index.html`
directly, because the page loads binary sidecar files:

```bash
cd "$OUT_DIR/full"
python -m http.server 8080 --bind 0.0.0.0
```

Expose port 8080 in Vast.ai, then open the URL for a per-call `index.html` path
listed in `analysis_report.json`.

## Outputs and interpretation

- `manifest.json` inventories reward details, expected tools, emitted tools,
  candidate tools, and selected verbose logs.
- `tool_scores.csv` contains one row per candidate and layer. A higher
  `mean_logprob` and lower `mean_rank` indicate stronger support. Use
  `sum_logprob` when comparing complete tool-name sequence likelihoods and
  remember that it penalizes longer names.
- `analysis_report.json` records per-call success, errors, and visualization
  paths.
- Each per-call directory contains an interactive Jacobian-lens slice.

Useful diagnostic patterns:

- An expected tool that is strong in middle layers but loses near the final
  layer suggests a late transformation or decision problem.
- An expected tool that is weak at every layer suggests that the relevant
  task or policy information was not represented strongly in this context.
- An expected tool that scores above the sampled tool at the model-final row
  points toward stochastic decoding or serving-time processing as a candidate
  explanation. The original run used non-greedy sampling, so this distinction
  matters.
- If expected and actual tool names match but tau2 still fails, inspect
  `reward_info` and arguments. Tool-name lens scores cannot diagnose a wrong
  argument, wrong action order, policy violation, or communication-evaluator
  failure by themselves.

These measurements are diagnostic correlations, not a causal proof. Use
matched success/failure runs and an intervention such as activation patching
before claiming a causal mechanism. More repetitions per task are much more
reliable than a single seed.
