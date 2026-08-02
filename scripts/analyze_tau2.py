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
    FailureEvent,
    LoggedCall,
    ReplayedResponse,
    SemanticBoundary,
    Tau2Case,
    build_tool_candidate,
    candidate_tool_names,
    discover_agent_calls,
    infer_first_error,
    load_cases,
    render_actual_response,
    select_calls,
    select_cases,
    select_event_calls,
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
    failure_event: FailureEvent | None = None
    anchor_index: int | None = None
    relative_to_failure: int | None = None
    reference_failures: tuple[str, ...] = ()
    next_call: LoggedCall | None = None


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


def _event_dict(event: FailureEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "call_index": event.call_index,
        "turn_index": event.turn_index,
        "kind": event.kind,
        "source": event.source,
        "confidence": event.confidence,
        "reason": event.reason,
    }


def load_failure_labels(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load manually audited first-error labels keyed by simulation id."""
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("labels"), list):
        value = value["labels"]
    if isinstance(value, list):
        labels: dict[str, dict[str, Any]] = {}
        for label in value:
            if not isinstance(label, dict) or label.get("simulation_id") is None:
                raise ValueError("every first-error label needs simulation_id")
            key = str(label["simulation_id"])
            if label.get("task_id") is not None:
                labels[f"{label['task_id']}/{key}"] = label
            labels[key] = label
        return labels
    if isinstance(value, dict) and all(isinstance(item, dict) for item in value.values()):
        return {str(key): item for key, item in value.items()}
    raise ValueError("first-error labels must be an object or a list of objects")


def _case_manifest(
    case: Tau2Case,
    all_calls: list[LoggedCall],
    selected_calls: list[LoggedCall],
    event: FailureEvent | None,
    reference_failures: tuple[str, ...],
) -> dict[str, Any]:
    candidates = []
    for call in selected_calls:
        candidates.extend(candidate_tool_names(case, call))
    selected_ids = {_call_id(call) for call in selected_calls}
    return {
        "task_id": case.task_id,
        "simulation_id": case.simulation_id,
        "outcome": "failure" if case.failed else "success",
        "reward": case.reward,
        "expected_tools": list(case.expected_tools),
        "actual_tools": list(case.actual_tools),
        "remaining_expected_tools": list(case.remaining_expected_tools),
        "reward_info": case.simulation.get("reward_info"),
        "first_error": _event_dict(event),
        "reference_failure_simulation_ids": list(reference_failures),
        "call_ids": [_call_id(call) for call in selected_calls],
        "calls": [
            {
                "call_id": _call_id(call),
                "call_index": call.call_index,
                "turn_index": call.turn_index,
                "selected": _call_id(call) in selected_ids,
                "path": str(call.path.resolve()),
                "model": (call.data.get("request") or {}).get("model"),
                "kwargs": (call.data.get("request") or {}).get("kwargs"),
                "tool_calls": (call.data.get("response") or {}).get("tool_calls"),
            }
            for call in all_calls
        ],
        "candidate_tools": list(dict.fromkeys(candidates)),
        "status": "ready" if all_calls else "missing_agent_logs",
    }


def build_manifest(
    run_dir: str | Path,
    *,
    include_successes: bool = False,
    pair_successes: bool = False,
    call_selection: Literal["last", "all", "event"] = "last",
    event_before: int | None = None,
    event_after: int = 1,
    failure_labels: dict[str, dict[str, Any]] | None = None,
    allow_review_labels: bool = True,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[SelectedCall]]:
    """Build a read-only inventory of cases and logs selected for analysis."""
    run_dir = Path(run_dir)
    saved_cases = load_cases(run_dir)
    labels = failure_labels or {}

    if call_selection == "event":
        failures = select_cases(saved_cases, include_successes=False, limit=limit)
        failure_tasks = {case.task_id for case in failures}
        successes = [case for case in saved_cases if not case.failed]
        if pair_successes and not include_successes:
            successes = [case for case in successes if case.task_id in failure_tasks]
        elif not include_successes:
            successes = []
        selected_ids = {case.simulation_id for case in [*failures, *successes]}
        cases = [case for case in saved_cases if case.simulation_id in selected_ids]
    else:
        cases = select_cases(
            saved_cases, include_successes=include_successes, limit=limit
        )

    calls_by_simulation = {
        case.simulation_id: discover_agent_calls(run_dir, case) for case in cases
    }
    events: dict[str, FailureEvent | None] = {}
    for case in cases:
        label = labels.get(f"{case.task_id}/{case.simulation_id}") or labels.get(
            case.simulation_id
        )
        events[case.simulation_id] = infer_first_error(
            case,
            calls_by_simulation[case.simulation_id],
            manual_label=label,
            allow_review_labels=allow_review_labels,
        )

    failures_by_task: dict[str, list[Tau2Case]] = {}
    for case in cases:
        if case.failed:
            failures_by_task.setdefault(case.task_id, []).append(case)

    selected: list[SelectedCall] = []
    manifest_cases = []
    missing_logs = 0
    for case in cases:
        all_calls = calls_by_simulation[case.simulation_id]
        event = events[case.simulation_id]
        reference_failures = tuple(
            failure.simulation_id
            for failure in failures_by_task.get(case.task_id, [])
            if failure.simulation_id != case.simulation_id
        )
        anchor: int | None = event.call_index if event is not None else None
        if call_selection == "event":
            if not case.failed:
                anchors = [
                    failure_event.call_index
                    for failure in failures_by_task.get(case.task_id, [])
                    if (
                        failure_event := events.get(failure.simulation_id)
                    ) is not None
                    and failure_event.call_index is not None
                ]
                anchor = anchors[0] if anchors else None
                selected_map: dict[int, LoggedCall] = {}
                if anchors:
                    for failure_anchor in anchors:
                        success_anchor = min(failure_anchor, max(0, len(all_calls) - 1))
                        for call in select_event_calls(
                            all_calls,
                            success_anchor,
                            before=event_before,
                            after=event_after,
                        ):
                            selected_map[call.call_index] = call
                else:
                    selected_map = {call.call_index: call for call in all_calls}
                calls = [selected_map[index] for index in sorted(selected_map)]
            else:
                calls = select_event_calls(
                    all_calls,
                    anchor,
                    before=event_before,
                    after=event_after,
                )
        else:
            calls = select_calls(all_calls, call_selection)

        manifest_cases.append(
            _case_manifest(case, all_calls, calls, event, reference_failures)
        )
        if not all_calls:
            missing_logs += 1
        for call in calls:
            next_call = (
                all_calls[call.call_index + 1]
                if call.call_index + 1 < len(all_calls)
                else None
            )
            selected.append(
                SelectedCall(
                    case=case,
                    call=call,
                    failure_event=event,
                    anchor_index=anchor,
                    relative_to_failure=(
                        call.call_index - anchor if anchor is not None else None
                    ),
                    reference_failures=reference_failures,
                    next_call=next_call,
                )
            )

    alignments: list[dict[str, Any]] = []
    successes_by_task: dict[str, list[Tau2Case]] = {}
    for case in cases:
        if not case.failed:
            successes_by_task.setdefault(case.task_id, []).append(case)
    for task_id, failures in failures_by_task.items():
        for failure in failures:
            event = events[failure.simulation_id]
            if event is None or event.call_index is None:
                continue
            failure_calls = calls_by_simulation[failure.simulation_id]
            start = 0 if event_before is None else max(0, event.call_index - event_before)
            stop = min(len(failure_calls), event.call_index + event_after + 1)
            for success in successes_by_task.get(task_id, []):
                success_calls = calls_by_simulation[success.simulation_id]
                if not success_calls:
                    continue
                success_anchor = min(event.call_index, len(success_calls) - 1)
                for failed_index in range(start, stop):
                    relative = failed_index - event.call_index
                    success_index = success_anchor + relative
                    if not 0 <= success_index < len(success_calls):
                        continue
                    alignments.append(
                        {
                            "task_id": task_id,
                            "failure_simulation_id": failure.simulation_id,
                            "success_simulation_id": success.simulation_id,
                            "relative_to_failure": relative,
                            "failure_call_id": _call_id(failure_calls[failed_index]),
                            "success_call_id": _call_id(success_calls[success_index]),
                            "failure_call_index": failed_index,
                            "success_call_index": success_index,
                        }
                    )

    manifest = {
        "run_dir": str(run_dir.resolve()),
        "selection": {
            "include_successes": include_successes,
            "pair_successes": pair_successes,
            "call_selection": call_selection,
            "event_before": "all" if event_before is None else event_before,
            "event_after": event_after,
            "allow_review_labels": allow_review_labels,
            "limit": limit,
        },
        "summary": {
            "saved_cases": len(saved_cases),
            "selected_cases": len(cases),
            "selected_calls": len(selected),
            "cases_without_agent_logs": missing_logs,
            "localized_failures": sum(
                event is not None and event.call_index is not None
                for event in events.values()
            ),
            "unlocalized_failures": sum(
                event is not None and event.call_index is None
                for event in events.values()
            ),
            "paired_success_cases": sum(not case.failed for case in cases),
            "alignment_rows": len(alignments),
        },
        "alignments": alignments,
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
        "call_index": selected.call.call_index,
        "turn_index": selected.call.turn_index,
        "outcome": "failure" if selected.case.failed else "success",
        "relative_to_failure": selected.relative_to_failure,
        "failure_kind": (
            selected.failure_event.kind if selected.failure_event is not None else None
        ),
        "failure_confidence": (
            selected.failure_event.confidence
            if selected.failure_event is not None
            else None
        ),
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


def _analysis_metadata(selected: SelectedCall) -> dict[str, Any]:
    return {
        "task_id": selected.case.task_id,
        "simulation_id": selected.case.simulation_id,
        "call_id": _call_id(selected.call),
        "call_index": selected.call.call_index,
        "turn_index": selected.call.turn_index,
        "outcome": "failure" if selected.case.failed else "success",
        "reward": selected.case.reward,
        "relative_to_failure": selected.relative_to_failure,
        "failure_kind": (
            selected.failure_event.kind if selected.failure_event is not None else None
        ),
        "failure_confidence": (
            selected.failure_event.confidence
            if selected.failure_event is not None
            else None
        ),
    }


def _decode_ids(tokenizer: Any, token_ids: list[int] | tuple[int, ...]) -> list[str]:
    return [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def _boundary_rows(
    selected: SelectedCall,
    *,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    rendered: Any,
    boundaries: list[SemanticBoundary],
    context_call_id: str,
    layers: list[int],
    max_seq_len: int,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if not boundaries:
        return []
    positions = [boundary.position for boundary in boundaries]
    exact_model = ExactInputModel(model, rendered.token_ids)
    lens_logits, model_logits, _ = lens.apply(
        exact_model,
        rendered.text,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
    )
    rows: list[dict[str, Any]] = []
    logits_by_layer: list[tuple[int | str, torch.Tensor]] = [
        *sorted(lens_logits.items()),
        ("model_final", model_logits),
    ]
    for layer, logits in logits_by_layer:
        count = min(top_k, logits.shape[-1])
        top_values, top_ids = torch.topk(logits, k=count, dim=-1)
        for row_index, boundary in enumerate(boundaries):
            token_ids = [int(token_id) for token_id in top_ids[row_index].tolist()]
            rows.append(
                {
                    **_analysis_metadata(selected),
                    "context_call_id": context_call_id,
                    "boundary": boundary.name,
                    "boundary_label": boundary.label,
                    "context": boundary.context,
                    "position": boundary.position,
                    "source_token_id": rendered.token_ids[boundary.position],
                    "source_token_piece": model.tokenizer.decode(
                        [rendered.token_ids[boundary.position]],
                        clean_up_tokenization_spaces=False,
                    ),
                    "layer": layer,
                    "top_token_ids": json.dumps(token_ids),
                    "top_token_pieces": json.dumps(
                        _decode_ids(model.tokenizer, token_ids), ensure_ascii=False
                    ),
                    "top_logits": json.dumps(
                        [float(value) for value in top_values[row_index].tolist()]
                    ),
                }
            )
    return rows


def score_semantic_boundaries(
    selected: SelectedCall,
    *,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    enable_thinking: bool,
    layer_stride: int,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], ReplayedResponse]:
    """Read out observation, decision, tool, argument, and update boundaries."""
    replay = render_actual_response(
        model.tokenizer,
        selected.call.data,
        enable_thinking=enable_thinking,
    )
    layers = _selected_layers(lens, layer_stride)
    rows: list[dict[str, Any]] = []
    request_boundaries = [
        boundary for boundary in replay.boundaries if boundary.context == "request"
    ]
    response_boundaries = [
        boundary for boundary in replay.boundaries if boundary.context == "response"
    ]
    rows.extend(
        _boundary_rows(
            selected,
            model=model,
            lens=lens,
            rendered=replay.request,
            boundaries=request_boundaries,
            context_call_id=_call_id(selected.call),
            layers=layers,
            max_seq_len=max_seq_len,
        )
    )
    rows.extend(
        _boundary_rows(
            selected,
            model=model,
            lens=lens,
            rendered=replay.response,
            boundaries=response_boundaries,
            context_call_id=_call_id(selected.call),
            layers=layers,
            max_seq_len=max_seq_len,
        )
    )
    if selected.next_call is not None:
        next_replay = render_actual_response(
            model.tokenizer,
            selected.next_call.data,
            enable_thinking=enable_thinking,
        )
        observation = next(
            boundary
            for boundary in next_replay.boundaries
            if boundary.name == "observation" and boundary.context == "request"
        )
        update = SemanticBoundary(
            name="update",
            context="update",
            position=observation.position,
            label=f"after {_call_id(selected.call)}",
        )
        rows.extend(
            _boundary_rows(
                selected,
                model=model,
                lens=lens,
                rendered=next_replay.request,
                boundaries=[update],
                context_call_id=_call_id(selected.next_call),
                layers=layers,
                max_seq_len=max_seq_len,
            )
        )
    return rows, replay


def score_generated_spans(
    selected: SelectedCall,
    replay: ReplayedResponse,
    *,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    layer_stride: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Score every logged tool-name and scalar argument-value token span."""
    spans = list(replay.generated_spans)
    if not spans:
        return []
    positions = [position for span in spans for position in span.prediction_positions]
    layers = _selected_layers(lens, layer_stride)
    exact_model = ExactInputModel(model, replay.response.token_ids)
    lens_logits, model_logits, _ = lens.apply(
        exact_model,
        replay.response.text,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for span in spans:
        width = len(span.token_ids)
        for layer, logits in [*sorted(lens_logits.items()), ("model_final", model_logits)]:
            score = summarize_logits(logits[offset : offset + width], span.token_ids)
            rows.append(
                {
                    **_analysis_metadata(selected),
                    "span_kind": span.kind,
                    "span_label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "layer": layer,
                    "token_count": width,
                    "sum_logprob": score.sum_logprob,
                    "mean_logprob": score.mean_logprob,
                    "mean_rank": score.mean_rank,
                    "max_rank": score.max_rank,
                    "token_ids": json.dumps(list(span.token_ids)),
                    "token_pieces": json.dumps(
                        _decode_ids(model.tokenizer, span.token_ids),
                        ensure_ascii=False,
                    ),
                    "ranks": json.dumps(list(score.ranks)),
                    "top_token_ids": json.dumps(list(score.top_token_ids)),
                    "top_token_pieces": json.dumps(
                        _decode_ids(model.tokenizer, score.top_token_ids),
                        ensure_ascii=False,
                    ),
                }
            )
        offset += width
    return rows


def write_visualization(
    selected: SelectedCall,
    *,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    replay: ReplayedResponse,
    output_dir: Path,
    pinned_token_ids: set[int],
    enable_thinking: bool,
    layer_stride: int,
    last_n_tokens: int | None,
    max_seq_len: int,
) -> Path:
    """Write the interactive slice including the logged assistant response."""
    del enable_thinking
    rendered = replay.response
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
            "Exact tau2 context with the logged assistant response teacher-forced. "
            "Pinned tokens come from expected and emitted tool names; boundary "
            "readouts are stored in semantic_boundary_readouts.csv."
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
    "call_index",
    "turn_index",
    "outcome",
    "relative_to_failure",
    "failure_kind",
    "failure_confidence",
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

BOUNDARY_FIELDS = [
    "task_id",
    "simulation_id",
    "call_id",
    "call_index",
    "turn_index",
    "outcome",
    "reward",
    "relative_to_failure",
    "failure_kind",
    "failure_confidence",
    "context_call_id",
    "boundary",
    "boundary_label",
    "context",
    "position",
    "source_token_id",
    "source_token_piece",
    "layer",
    "top_token_ids",
    "top_token_pieces",
    "top_logits",
]

GENERATED_SPAN_FIELDS = [
    "task_id",
    "simulation_id",
    "call_id",
    "call_index",
    "turn_index",
    "outcome",
    "reward",
    "relative_to_failure",
    "failure_kind",
    "failure_confidence",
    "span_kind",
    "span_label",
    "start",
    "end",
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


def _write_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def _parse_event_before(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use 'all' or a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("event window must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    parser.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    parser.add_argument(
        "--include-successes",
        action="store_true",
        help="include every saved success, not only task-matched pairs",
    )
    parser.add_argument(
        "--pair-successes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include successful runs of the same task for matched comparison",
    )
    parser.add_argument(
        "--call-selection", choices=("last", "all", "event"), default="event"
    )
    parser.add_argument(
        "--event-before",
        type=_parse_event_before,
        default=None,
        metavar="N|all",
        help="calls before t_fail to retain; default 'all' keeps full history",
    )
    parser.add_argument("--event-after", type=int, default=1)
    parser.add_argument(
        "--first-error-labels",
        type=Path,
        help="manually audited JSON labels keyed by simulation id",
    )
    parser.add_argument(
        "--allow-review-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use tau2 reviewer turn labels when no verified checker fires",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument(
        "--last-n-tokens",
        type=int,
        default=96,
        help="HTML slice width; 0 renders all tokens (boundary CSV is unaffected)",
    )
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.event_after < 0:
        raise ValueError("--event-after must be non-negative")
    if args.last_n_tokens < 0:
        raise ValueError("--last-n-tokens must be non-negative")
    output_dir = args.output_dir or args.run_dir / "jlens-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_labels = load_failure_labels(args.first_error_labels)
    manifest, selected_calls = build_manifest(
        args.run_dir,
        include_successes=args.include_successes,
        pair_successes=args.pair_successes,
        call_selection=args.call_selection,
        event_before=args.event_before,
        event_after=args.event_after,
        failure_labels=failure_labels,
        allow_review_labels=args.allow_review_labels,
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
    all_boundary_rows: list[dict[str, Any]] = []
    all_generated_span_rows: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_calls, start=1):
        label = (
            f"task={selected.case.task_id} sim={selected.case.simulation_id} "
            f"call={_call_id(selected.call)}"
        )
        print(f"[{index}/{len(selected_calls)}] {label}")
        entry: dict[str, Any] = {"label": label, "status": "ok"}
        try:
            boundary_rows, replay = score_semantic_boundaries(
                selected,
                model=model,
                lens=lens,
                enable_thinking=args.enable_thinking,
                layer_stride=args.layer_stride,
                max_seq_len=args.max_seq_len,
            )
            generated_span_rows = score_generated_spans(
                selected,
                replay,
                model=model,
                lens=lens,
                layer_stride=args.layer_stride,
                max_seq_len=args.max_seq_len,
            )
            all_boundary_rows.extend(boundary_rows)
            all_generated_span_rows.extend(generated_span_rows)
            entry["boundary_rows"] = len(boundary_rows)
            entry["generated_span_rows"] = len(generated_span_rows)
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
                        replay=replay,
                        output_dir=call_dir,
                        pinned_token_ids=pinned,
                        enable_thinking=args.enable_thinking,
                        layer_stride=args.layer_stride,
                        last_n_tokens=(
                            None if args.last_n_tokens == 0 else args.last_n_tokens
                        ),
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
    boundary_path = output_dir / "semantic_boundary_readouts.csv"
    _write_rows(boundary_path, all_boundary_rows, BOUNDARY_FIELDS)
    generated_span_path = output_dir / "generated_span_scores.csv"
    _write_rows(
        generated_span_path, all_generated_span_rows, GENERATED_SPAN_FIELDS
    )
    report_path = output_dir / "analysis_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(all_rows)} score rows to {scores_path}")
    print(
        f"Wrote {len(all_boundary_rows)} semantic-boundary rows to {boundary_path}"
    )
    print(
        f"Wrote {len(all_generated_span_rows)} generated-span rows to "
        f"{generated_span_path}"
    )
    print(f"Per-call status: {report_path}")


if __name__ == "__main__":
    main()
