# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Adapters for replaying tau2 agent calls through a Jacobian lens.

The tau2 ``results.json`` file supplies task labels and rewards, while verbose
LLM logs contain the exact messages and tool schemas sent to the agent.  This
module deliberately has no dependency on tau2 so a saved run can be analysed
in the lightweight Jacobian-lens environment.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import torch

JSONDict = dict[str, Any]
AGENT_CALL_NAMES = frozenset(
    {"agent_response", "agent_gt_response", "agent_solo_response"}
)


@dataclass(frozen=True)
class Tau2Case:
    """One saved tau2 simulation and the labels needed for analysis."""

    task_id: str
    simulation_id: str
    reward: float | None
    expected_tools: tuple[str, ...]
    actual_tools: tuple[str, ...]
    task: JSONDict
    simulation: JSONDict

    @property
    def failed(self) -> bool:
        """Whether tau2 assigned a non-perfect reward."""
        return self.reward is not None and self.reward < 1.0

    @property
    def remaining_expected_tools(self) -> tuple[str, ...]:
        """Expected assistant actions not already present in the trajectory."""
        actual_counts = Counter(self.actual_tools)
        remaining: list[str] = []
        for name in self.expected_tools:
            if actual_counts[name]:
                actual_counts[name] -= 1
            else:
                remaining.append(name)
        return tuple(remaining)


@dataclass(frozen=True)
class LoggedCall:
    """A verbose tau2 LLM log for one agent response."""

    path: Path
    data: JSONDict

    @property
    def actual_tools(self) -> tuple[str, ...]:
        response = self.data.get("response") or {}
        return _tool_names(response.get("tool_calls") or [])


@dataclass(frozen=True)
class RenderedCall:
    """A logged request rendered with the model's Hugging Face chat template."""

    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ToolCandidate:
    """Teacher-forced tool name and positions whose logits predict its tokens."""

    name: str
    text: str
    token_ids: tuple[int, ...]
    name_token_ids: tuple[int, ...]
    name_start: int
    prediction_positions: tuple[int, ...]


@dataclass(frozen=True)
class LogitScore:
    """Aggregate and per-token scores for one teacher-forced candidate."""

    sum_logprob: float
    mean_logprob: float
    ranks: tuple[int, ...]
    mean_rank: float
    max_rank: int
    top_token_ids: tuple[int, ...]


def _read_json(path: Path) -> JSONDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _results_file(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return path
    return path / "results.json"


def _saved_simulations(results_file: Path, results: JSONDict) -> list[JSONDict]:
    simulations = results.get("simulations") or []
    if simulations:
        return [
            simulation for simulation in simulations if isinstance(simulation, dict)
        ]

    simulations_dir = results_file.parent / "simulations"
    if not simulations_dir.is_dir():
        return []
    return [_read_json(path) for path in sorted(simulations_dir.glob("*.json"))]


def _reward_of(simulation: Mapping[str, Any]) -> float | None:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward") if isinstance(reward_info, Mapping) else None
    if reward is None and isinstance(simulation.get("reward"), (int, float)):
        reward = simulation["reward"]
    return float(reward) if isinstance(reward, (int, float)) else None


def _tool_name(tool_call: Mapping[str, Any]) -> str | None:
    name = tool_call.get("name")
    if isinstance(name, str):
        return name
    function = tool_call.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _tool_names(tool_calls: Sequence[Any]) -> tuple[str, ...]:
    names = []
    for tool_call in tool_calls:
        if isinstance(tool_call, Mapping) and (name := _tool_name(tool_call)):
            names.append(name)
    return tuple(names)


def _actual_tools(simulation: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for message in simulation.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        names.extend(_tool_names(message.get("tool_calls") or []))
    return tuple(names)


def _expected_tools(task: Mapping[str, Any]) -> tuple[str, ...]:
    criteria = task.get("evaluation_criteria") or {}
    if not isinstance(criteria, Mapping):
        return ()
    actions = criteria.get("actions") or []
    names: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        if action.get("requestor", "assistant") != "assistant":
            continue
        name = action.get("name")
        if isinstance(name, str):
            names.append(name)
    return tuple(names)


def load_cases(path: str | Path) -> list[Tau2Case]:
    """Load text simulations from a tau2 result file or run directory."""
    results_file = _results_file(path)
    results = _read_json(results_file)
    tasks = {
        str(task["id"]): task
        for task in results.get("tasks") or []
        if isinstance(task, dict) and "id" in task
    }

    cases = []
    for simulation in _saved_simulations(results_file, results):
        task_id = str(simulation.get("task_id"))
        simulation_id = str(simulation.get("id"))
        task = tasks.get(task_id, {})
        cases.append(
            Tau2Case(
                task_id=task_id,
                simulation_id=simulation_id,
                reward=_reward_of(simulation),
                expected_tools=_expected_tools(task),
                actual_tools=_actual_tools(simulation),
                task=task,
                simulation=simulation,
            )
        )
    return cases


def select_cases(
    cases: Sequence[Tau2Case],
    *,
    include_successes: bool = False,
    limit: int | None = None,
) -> list[Tau2Case]:
    """Select failed cases by default, preserving the saved run order."""
    selected = (
        list(cases) if include_successes else [case for case in cases if case.failed]
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]
    return selected


CallT = TypeVar("CallT")


def select_calls(
    calls: Sequence[CallT], selection: Literal["last", "all"]
) -> list[CallT]:
    """Select all logged calls or only the final call for a simulation."""
    if selection == "all":
        return list(calls)
    if selection == "last":
        return list(calls[-1:])
    raise ValueError(f"unknown call selection {selection!r}")


def candidate_tool_names(case: Tau2Case, call: LoggedCall) -> tuple[str, ...]:
    """Return unmet expected tools followed by tools emitted in this call."""
    expected = case.remaining_expected_tools or case.expected_tools
    return tuple(dict.fromkeys((*expected, *call.actual_tools)))


def discover_agent_calls(run_dir: str | Path, case: Tau2Case) -> list[LoggedCall]:
    """Find verbose agent-call logs associated with ``case``."""
    log_dir = (
        Path(run_dir)
        / "artifacts"
        / f"task_{case.task_id}"
        / f"sim_{case.simulation_id}"
        / "llm_debug"
    )
    calls = []
    for path in sorted(log_dir.glob("*.json")):
        data = _read_json(path)
        if data.get("call_name") in AGENT_CALL_NAMES:
            calls.append(LoggedCall(path=path, data=data))
    return calls


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[JSONDict]:
    """Undo tau2's newline-splitting transformation without mutating input."""
    normalized: list[JSONDict] = []
    for message in messages:
        value = copy.deepcopy(dict(message))
        if isinstance(value.get("content"), list):
            value["content"] = "\n".join(str(line) for line in value["content"])
        normalized.append(value)
    return normalized


def _chat_template_kwargs(call: Mapping[str, Any], enable_thinking: bool) -> JSONDict:
    request = call.get("request") or {}
    messages = request.get("messages") or []
    kwargs: JSONDict = {
        "conversation": normalize_messages(messages),
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if request.get("tools"):
        kwargs["tools"] = request["tools"]
    return kwargs


def _as_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError("chat template did not return a flat token-id list")
    return tuple(value)


def render_logged_call(
    tokenizer: Any, call: Mapping[str, Any], *, enable_thinking: bool
) -> RenderedCall:
    """Render the exact pre-response context recorded by tau2."""
    kwargs = _chat_template_kwargs(call, enable_thinking)
    text = tokenizer.apply_chat_template(**kwargs)
    token_ids = tokenizer.apply_chat_template(**{**kwargs, "tokenize": True})
    if not isinstance(text, str):
        raise TypeError("chat template did not return text")
    return RenderedCall(text=text, token_ids=_as_token_ids(token_ids))


def _find_subsequence(
    values: Sequence[int], target: Sequence[int], *, start: int = 0
) -> int:
    if not target:
        raise ValueError("cannot locate an empty token sequence")
    stop = len(values) - len(target) + 1
    for index in range(start, stop):
        if tuple(values[index : index + len(target)]) == tuple(target):
            return index
    raise ValueError("tool name was not found in the rendered assistant tool call")


def build_tool_candidate(
    tokenizer: Any,
    call: Mapping[str, Any],
    tool_name: str,
    *,
    enable_thinking: bool,
) -> ToolCandidate:
    """Render a teacher-forced assistant tool call for layer-wise scoring."""
    request = call.get("request") or {}
    messages = normalize_messages(request.get("messages") or [])
    candidate_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": tool_name, "arguments": {}},
            }
        ],
    }
    kwargs: JSONDict = {
        "conversation": [*messages, candidate_message],
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": enable_thinking,
    }
    if request.get("tools"):
        kwargs["tools"] = request["tools"]

    text = tokenizer.apply_chat_template(**kwargs)
    token_ids = _as_token_ids(
        tokenizer.apply_chat_template(**{**kwargs, "tokenize": True})
    )
    name_token_ids = _as_token_ids(
        tokenizer.encode(tool_name, add_special_tokens=False)
    )
    base_length = len(
        render_logged_call(tokenizer, call, enable_thinking=enable_thinking).token_ids
    )
    name_start = _find_subsequence(
        token_ids, name_token_ids, start=max(0, base_length - 1)
    )
    name_end = name_start + len(name_token_ids)
    prediction_positions = tuple(range(name_start - 1, name_end - 1))
    if prediction_positions[0] < 0:
        raise ValueError("tool name has no preceding token to score")
    if not isinstance(text, str):
        raise TypeError("chat template did not return text")
    return ToolCandidate(
        name=tool_name,
        text=text,
        token_ids=token_ids,
        name_token_ids=name_token_ids,
        name_start=name_start,
        prediction_positions=prediction_positions,
    )


def summarize_logits(logits: torch.Tensor, target_ids: Sequence[int]) -> LogitScore:
    """Score one target token at each row of a ``[tokens, vocab]`` tensor."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocab]")
    if logits.shape[0] != len(target_ids) or not target_ids:
        raise ValueError("target_ids must contain one token id per logits row")

    targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    row = torch.arange(len(target_ids), device=logits.device)
    target_logits = logits[row, targets]
    token_logprobs = target_logits - torch.logsumexp(logits, dim=-1)
    ranks = (logits > target_logits[:, None]).sum(dim=-1) + 1
    top_token_ids = logits.argmax(dim=-1)
    return LogitScore(
        sum_logprob=float(token_logprobs.sum().item()),
        mean_logprob=float(token_logprobs.mean().item()),
        ranks=tuple(int(rank) for rank in ranks.tolist()),
        mean_rank=float(ranks.float().mean().item()),
        max_rank=int(ranks.max().item()),
        top_token_ids=tuple(int(token_id) for token_id in top_token_ids.tolist()),
    )
