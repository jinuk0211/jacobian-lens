import json

import pytest
import torch

from jlens.tau2 import (
    build_tool_candidate,
    candidate_tool_names,
    discover_agent_calls,
    load_cases,
    normalize_messages,
    render_logged_call,
    select_calls,
    select_cases,
    summarize_logits,
)


class FakeTokenizer:
    """Small character tokenizer with a deterministic tool-call template."""

    @staticmethod
    def encode(text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    @staticmethod
    def decode(token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        conversation,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        del tools, kwargs
        rendered = ""
        for message in conversation:
            rendered += f"<{message['role']}>"
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                name = tool_calls[0]["function"]["name"]
                rendered += f'<tool_call>{{"name":"{name}"}}'
            else:
                rendered += message.get("content") or ""
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered)
        return rendered


class MappingTokenizer(FakeTokenizer):
    """Matches the BatchEncoding shape returned by Transformers 5."""

    def apply_chat_template(self, *args, **kwargs):
        value = super().apply_chat_template(*args, **kwargs)
        return {"input_ids": value} if kwargs.get("tokenize") else value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sample_call():
    return {
        "call_id": "call-1",
        "call_name": "agent_response",
        "timestamp": "2026-07-31T14:13:17",
        "request": {
            "model": "hosted_vllm/Qwen3-8B",
            "messages": [
                {"role": "system", "content": ["policy line 1", "line 2"]},
                {"role": "user", "content": "cancel my flight"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_reservation",
                        "description": "Cancel a reservation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "response": {
            "content": None,
            "tool_calls": [
                {
                    "id": "tool-1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": "ABC123"},
                }
            ],
        },
    }


def test_load_cases_and_discover_agent_logs(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "results.json",
        {
            "tasks": [
                {
                    "id": "5",
                    "evaluation_criteria": {
                        "actions": [
                            {
                                "requestor": "assistant",
                                "name": "get_reservation_details",
                            },
                            {
                                "requestor": "assistant",
                                "name": "cancel_reservation",
                            },
                            {"requestor": "user", "name": "dismiss_notification"},
                        ]
                    },
                }
            ],
            "simulations": [
                {
                    "id": "sim-5",
                    "task_id": "5",
                    "reward_info": {"reward": 0.0},
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "name": "get_reservation_details",
                                    "arguments": {"reservation_id": "ABC123"},
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    log_dir = run_dir / "artifacts" / "task_5" / "sim_sim-5" / "llm_debug"
    write_json(log_dir / "001_agent_response.json", sample_call())
    write_json(
        log_dir / "002_user_response.json",
        {**sample_call(), "call_id": "call-2", "call_name": "user_response"},
    )

    cases = load_cases(run_dir)

    assert len(cases) == 1
    assert cases[0].reward == 0.0
    assert cases[0].expected_tools == (
        "get_reservation_details",
        "cancel_reservation",
    )
    assert cases[0].actual_tools == ("get_reservation_details",)
    assert cases[0].remaining_expected_tools == ("cancel_reservation",)

    calls = discover_agent_calls(run_dir, cases[0])
    assert len(calls) == 1
    assert calls[0].actual_tools == ("get_reservation_details",)
    assert calls[0].path.name == "001_agent_response.json"


def test_load_cases_supports_directory_storage(tmp_path):
    run_dir = tmp_path / "run"
    write_json(run_dir / "results.json", {"tasks": [{"id": "1"}]})
    write_json(
        run_dir / "simulations" / "sim-1.json",
        {"id": "sim-1", "task_id": "1", "reward_info": {"reward": 1.0}},
    )

    cases = load_cases(run_dir)

    assert [(case.simulation_id, case.reward) for case in cases] == [("sim-1", 1.0)]


def test_normalize_messages_restores_logged_multiline_content():
    messages = sample_call()["request"]["messages"]

    normalized = normalize_messages(messages)

    assert normalized[0]["content"] == "policy line 1\nline 2"
    assert messages[0]["content"] == ["policy line 1", "line 2"]


def test_render_and_tool_candidate_use_the_chat_template():
    tokenizer = FakeTokenizer()
    call = sample_call()

    rendered = render_logged_call(tokenizer, call, enable_thinking=False)
    candidate = build_tool_candidate(
        tokenizer,
        call,
        "cancel_reservation",
        enable_thinking=False,
    )

    assert rendered.text.endswith("<assistant>")
    assert rendered.token_ids == tuple(tokenizer.encode(rendered.text))
    assert candidate.name_token_ids == tuple(tokenizer.encode("cancel_reservation"))
    name_end = candidate.name_start + len(candidate.name_token_ids)
    assert (
        candidate.token_ids[candidate.name_start : name_end] == candidate.name_token_ids
    )
    assert candidate.prediction_positions == tuple(
        range(candidate.name_start - 1, name_end - 1)
    )


def test_render_accepts_transformers_five_batch_encoding_shape():
    tokenizer = MappingTokenizer()

    rendered = render_logged_call(tokenizer, sample_call(), enable_thinking=False)

    assert rendered.token_ids == tuple(tokenizer.encode(rendered.text))


def test_select_failures_last_call_and_candidate_tools(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "results.json",
        {
            "tasks": [
                {
                    "id": "5",
                    "evaluation_criteria": {
                        "actions": [
                            {"name": "lookup_booking"},
                            {"name": "cancel_reservation"},
                        ]
                    },
                },
                {"id": "6", "evaluation_criteria": {"actions": []}},
            ],
            "simulations": [
                {
                    "id": "failed",
                    "task_id": "5",
                    "reward_info": {"reward": 0.0},
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [{"name": "lookup_booking"}],
                        }
                    ],
                },
                {
                    "id": "passed",
                    "task_id": "6",
                    "reward_info": {"reward": 1.0},
                },
            ],
        },
    )
    case = select_cases(load_cases(run_dir))[0]
    calls = [
        type("Call", (), {"actual_tools": ("lookup_booking",)})(),
        type("Call", (), {"actual_tools": ("search_flights",)})(),
    ]

    assert case.simulation_id == "failed"
    assert select_calls(calls, "last") == [calls[-1]]
    assert select_calls(calls, "all") == calls
    assert candidate_tool_names(case, calls[-1]) == (
        "cancel_reservation",
        "search_flights",
    )
    assert len(select_cases(load_cases(run_dir), include_successes=True)) == 2


def test_summarize_logits_uses_one_based_ranks_and_length_normalized_score():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]])

    score = summarize_logits(logits, (1, 2))

    expected = torch.log_softmax(logits, dim=-1)[[0, 1], [1, 2]]
    assert score.sum_logprob == pytest.approx(expected.sum().item())
    assert score.mean_logprob == pytest.approx(expected.mean().item())
    assert score.ranks == (1, 3)
    assert score.mean_rank == 2.0
    assert score.max_rank == 3
    assert score.top_token_ids == (1, 0)
