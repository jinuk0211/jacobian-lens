#!/usr/bin/env python
"""Replay saved tau2 agent calls through a pre-fitted Jacobian lens."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

import jlens
from jlens.tau2 import (
    LoggedCall,
    Tau2Case,
    build_tool_candidate,
    candidate_tool_names,
    discover_agent_calls,
    load_cases,
    render_logged_call,
    select_calls,
    select_cases,
    summarize_logits,
)
from jlens.vis import build_page, compute_slice

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_REVISION = "91271eb5b15a43eebed7bb447618738754f1379a"
DEFAULT_LENS_FILE = "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"


@dataclass(frozen=True)
class SelectedCall:
    """A simulation paired with one verbose agent-response log."""

    case: Tau2Case
    call: LoggedCall


class ExactInputModel:
    """Lens model wrapper that returns pre-rendered chat-template token IDs."""

    def __init__(self, model: jlens.LensModel, token_ids: tuple[int, ...]) -> None:
        self._model = model
        self._token_ids = token_ids
        self.tokenizer = model.tokenizer
        self.n_layers = model.n_layers
        self.d_model = model.d_model
        self.layers = model.layers

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        del text
        if len(self._token_ids) > max_length:
            raise ValueError(
                f"rendered request has {len(self._token_ids)} tokens, exceeding "
                f"--max-seq-len={max_length}; increase the limit explicitly"
            )
        return torch.tensor(
            [self._token_ids], dtype=torch.long, device=self._model.input_device
        )

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self._model.forward(input_ids)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self._model.unembed(residual)


def _call_id(call: LoggedCall) -> str:
    value = call.data.get("call_id")
    return str(value) if value is not None else call.path.stem


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _case_manifest(case: Tau2Case, calls: list[LoggedCall]) -> dict[str, Any]:
    candidates = []
    for call in calls:
        candidates.extend(candidate_tool_names(case, call))
    return {
        "task_id": case.task_id,
        "simulation_id": case.simulation_id,
        "reward": case.reward,
        "expected_tools": list(case.expected_tools),
        "actual_tools": list(case.actual_tools),
        "remaining_expected_tools": list(case.remaining_expected_tools),
        "reward_info": case.simulation.get("reward_info"),
        "call_ids": [_call_id(call) for call in calls],
        "calls": [
            {
                "call_id": _call_id(call),
                "path": str(call.path.resolve()),
                "model": (call.data.get("request") or {}).get("model"),
                "kwargs": (call.data.get("request") or {}).get("kwargs"),
                "tool_calls": (call.data.get("response") or {}).get("tool_calls"),
            }
            for call in calls
        ],
        "candidate_tools": list(dict.fromkeys(candidates)),
        "status": "ready" if calls else "missing_agent_logs",
    }


def build_manifest(
    run_dir: str | Path,
    *,
    include_successes: bool = False,
    call_selection: Literal["last", "all"] = "last",
    limit: int | None = None,
) -> tuple[dict[str, Any], list[SelectedCall]]:
    """Build a read-only inventory of cases and logs selected for analysis."""
    run_dir = Path(run_dir)
    saved_cases = load_cases(run_dir)
    cases = select_cases(saved_cases, include_successes=include_successes, limit=limit)
    selected: list[SelectedCall] = []
    manifest_cases = []
    missing_logs = 0
    for case in cases:
        calls = select_calls(discover_agent_calls(run_dir, case), call_selection)
        manifest_cases.append(_case_manifest(case, calls))
        if not calls:
            missing_logs += 1
        selected.extend(SelectedCall(case=case, call=call) for call in calls)

    manifest = {
        "run_dir": str(run_dir.resolve()),
        "selection": {
            "include_successes": include_successes,
            "call_selection": call_selection,
            "limit": limit,
        },
        "summary": {
            "saved_cases": len(saved_cases),
            "selected_cases": len(cases),
            "selected_calls": len(selected),
            "cases_without_agent_logs": missing_logs,
        },
        "cases": manifest_cases,
    }
    return manifest, selected


def _selected_layers(lens: jlens.JacobianLens, stride: int) -> list[int]:
    if stride < 1:
        raise ValueError("--layer-stride must be at least 1")
    layers = lens.source_layers[::stride]
    if lens.source_layers and lens.source_layers[-1] not in layers:
        layers.append(lens.source_layers[-1])
    return layers


def _candidate_kind(case: Tau2Case, call: LoggedCall, name: str) -> str:
    expected = name in case.remaining_expected_tools or name in case.expected_tools
    actual = name in call.actual_tools
    if expected and actual:
        return "expected_and_actual"
    if expected:
        return "expected"
    return "actual"


def _score_row(
    *,
    selected: SelectedCall,
    candidate_name: str,
    layer: int | str,
    logits: torch.Tensor,
    target_ids: tuple[int, ...],
    tokenizer: Any,
) -> dict[str, Any]:
    score = summarize_logits(logits, target_ids)
    return {
        "task_id": selected.case.task_id,
        "simulation_id": selected.case.simulation_id,
        "call_id": _call_id(selected.call),
        "reward": selected.case.reward,
        "candidate_tool": candidate_name,
        "candidate_kind": _candidate_kind(selected.case, selected.call, candidate_name),
        "layer": layer,
        "token_count": len(target_ids),
        "sum_logprob": score.sum_logprob,
        "mean_logprob": score.mean_logprob,
        "mean_rank": score.mean_rank,
        "max_rank": score.max_rank,
        "token_ids": json.dumps(list(target_ids)),
        "token_pieces": json.dumps(
            [
                tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                for token_id in target_ids
            ],
            ensure_ascii=False,
        ),
        "ranks": json.dumps(list(score.ranks)),
        "top_token_ids": json.dumps(list(score.top_token_ids)),
        "top_token_pieces": json.dumps(
            [
                tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                for token_id in score.top_token_ids
            ],
            ensure_ascii=False,
        ),
    }


def score_call(
    selected: SelectedCall,
    *,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    enable_thinking: bool,
    layer_stride: int,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Score expected and actual tool names for one saved agent call."""
    rows: list[dict[str, Any]] = []
    pinned_token_ids: set[int] = set()
    layers = _selected_layers(lens, layer_stride)
    for name in candidate_tool_names(selected.case, selected.call):
        candidate = build_tool_candidate(
            model.tokenizer,
            selected.call.data,
            name,
            enable_thinking=enable_thinking,
        )
        pinned_token_ids.update(candidate.name_token_ids)
        exact_model = ExactInputModel(model, candidate.token_ids)
        lens_logits, model_logits, _ = lens.apply(
            exact_model,
            candidate.text,
            layers=layers,
            positions=candidate.prediction_positions,
            max_seq_len=max_seq_len,
        )
        for layer, logits in sorted(lens_logits.items()):
            rows.append(
                _score_row(
                    selected=selected,
                    candidate_name=name,
                    layer=layer,
                    logits=logits,
                    target_ids=candidate.name_token_ids,
                    tokenizer=model.tokenizer,
                )
            )
        rows.append(
            _score_row(
                selected=selected,
                candidate_name=name,
                layer="model_final",
                logits=model_logits,
                target_ids=candidate.name_token_ids,
                tokenizer=model.tokenizer,
            )
        )
    return rows, pinned_token_ids


def write_visualization(
    selected: SelectedCall,
    *,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    output_dir: Path,
    pinned_token_ids: set[int],
    enable_thinking: bool,
    layer_stride: int,
    last_n_tokens: int,
    max_seq_len: int,
) -> Path:
    """Write the interactive context slice for one saved request."""
    rendered = render_logged_call(
        model.tokenizer,
        selected.call.data,
        enable_thinking=enable_thinking,
    )
    exact_model = ExactInputModel(model, rendered.token_ids)
    slice_data = compute_slice(
        exact_model,
        lens,
        rendered.text,
        top_n=5,
        max_tracked=32,
        pinned_token_ids=pinned_token_ids,
        layer_stride=layer_stride,
        last_n_tokens=last_n_tokens,
        max_seq_len=max_seq_len,
    )
    page, _, _ = build_page(
        slice_data,
        rendered.text,
        title=(
            f"tau2 task {selected.case.task_id} / "
            f"simulation {selected.case.simulation_id}"
        ),
        description=(
            "Exact pre-response context from tau2 verbose logs. Pinned tokens "
            "come from expected and emitted tool names."
        ),
        pinned_token_ids=pinned_token_ids,
        mode="fetch",
        out_dir=output_dir,
    )
    html_path = output_dir / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


SCORE_FIELDS = [
    "task_id",
    "simulation_id",
    "call_id",
    "reward",
    "candidate_tool",
    "candidate_kind",
    "layer",
    "token_count",
    "sum_logprob",
    "mean_logprob",
    "mean_rank",
    "max_rank",
    "token_ids",
    "token_pieces",
    "ranks",
    "top_token_ids",
    "top_token_pieces",
]


def _write_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_model_and_lens(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-8B analysis")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).cuda()
    model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_file,
        revision=args.lens_revision,
    )
    if lens.d_model != model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} does not match model d_model={model.d_model}"
        )
    return hf_model, model, lens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    parser.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    parser.add_argument("--include-successes", action="store_true")
    parser.add_argument("--call-selection", choices=("last", "all"), default="last")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--last-n-tokens", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "jlens-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, selected_calls = build_manifest(
        args.run_dir,
        include_successes=args.include_successes,
        call_selection=args.call_selection,
        limit=args.limit,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = manifest["summary"]
    print(
        f"Selected {summary['selected_calls']} agent calls from "
        f"{summary['selected_cases']} cases; manifest: {manifest_path}"
    )
    if args.inspect_only:
        return
    if not selected_calls:
        raise RuntimeError(
            "no agent logs selected; rerun tau2 with --verbose-logs "
            "--llm-log-mode all and check the run directory"
        )

    _, model, lens = _load_model_and_lens(args)
    all_rows: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_calls, start=1):
        label = (
            f"task={selected.case.task_id} sim={selected.case.simulation_id} "
            f"call={_call_id(selected.call)}"
        )
        print(f"[{index}/{len(selected_calls)}] {label}")
        entry: dict[str, Any] = {"label": label, "status": "ok"}
        try:
            rows, pinned = score_call(
                selected,
                model=model,
                lens=lens,
                enable_thinking=args.enable_thinking,
                layer_stride=args.layer_stride,
                max_seq_len=args.max_seq_len,
            )
            all_rows.extend(rows)
            entry["score_rows"] = len(rows)
            if not args.no_html:
                call_dir = output_dir / (
                    f"task_{_safe_name(selected.case.task_id)}__"
                    f"sim_{_safe_name(selected.case.simulation_id)}__"
                    f"call_{_safe_name(_call_id(selected.call))}"
                )
                entry["visualization"] = str(
                    write_visualization(
                        selected,
                        model=model,
                        lens=lens,
                        output_dir=call_dir,
                        pinned_token_ids=pinned,
                        enable_thinking=args.enable_thinking,
                        layer_stride=args.layer_stride,
                        last_n_tokens=args.last_n_tokens,
                        max_seq_len=args.max_seq_len,
                    )
                )
        except (RuntimeError, TypeError, ValueError) as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"  error: {exc}")
        report.append(entry)

    scores_path = output_dir / "tool_scores.csv"
    _write_scores(scores_path, all_rows)
    report_path = output_dir / "analysis_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(all_rows)} score rows to {scores_path}")
    print(f"Per-call status: {report_path}")


if __name__ == "__main__":
    main()
