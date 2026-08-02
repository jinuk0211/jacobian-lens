import json

import pytest
import torch

from jlens.tau2 import (
    LoggedCall,
    Tau2Case,
    build_tool_candidate,
    candidate_tool_names,
    discover_agent_calls,
    infer_first_error,
    load_cases,
    normalize_messages,
    render_actual_response,
    render_logged_call,
    select_calls,
    select_cases,
    select_event_calls,
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
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    name = function["name"]
                    arguments = json.dumps(
                        function.get("arguments") or {}, separators=(",", ":")
                    )
                    rendered += f'<tool_call>{{"name":"{name}","arguments":{arguments}}}'
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


class ContextMergingTokenizer:
    """Qwen3.5-like XML template whose tool-name token includes its context."""

    def __init__(self):
        self._piece_to_id = {}
        self._id_to_piece = {}

    def _piece_id(self, piece):
        if piece not in self._piece_to_id:
            token_id = 256 + len(self._piece_to_id)
            self._piece_to_id[piece] = token_id
            self._id_to_piece[token_id] = piece
        return self._piece_to_id[piece]

    def _tokenize(self, text):
        pieces = []
        index = 0
        while index < len(text):
            if text.startswith("<function=", index):
                end = text.index(">", index) + 1
                pieces.append((text[index:end], index, end))
                index = end
            else:
                pieces.append((text[index], index, index + 1))
                index += 1
        return pieces

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [self._piece_id(piece) for piece, _, _ in self._tokenize(text)]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(self._id_to_piece[token_id] for token_id in token_ids)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        pieces = self._tokenize(text)
        value = {"input_ids": [self._piece_id(piece) for piece, _, _ in pieces]}
        if return_offsets_mapping:
            value["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return value

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
            rendered += f"<{message['role']}>" + (message.get("content") or "")
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or tool_call
                arguments = function.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise TypeError("Can only get item pairs from a mapping.")
                rendered += f"<tool_call><function={function['name']}>"
                for name, value in arguments.items():
                    rendered += f"<parameter={name}>{value}</parameter>"
                rendered += "</function></tool_call>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered) if tokenize else rendered


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


def test_normalize_messages_parses_openai_tool_argument_strings():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_booking",
                        "arguments": '{"booking_id":"ABC123"}',
                    },
                }
            ],
        }
    ]

    normalized = normalize_messages(messages)

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
        "booking_id": "ABC123"
    }
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"booking_id":"ABC123"}'
    )


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


def test_render_actual_response_locates_semantic_boundaries_and_arguments():
    replay = render_actual_response(
        FakeTokenizer(), sample_call(), enable_thinking=False
    )

    assert [boundary.name for boundary in replay.boundaries] == [
        "observation",
        "decision",
        "tool",
        "argument",
    ]
    assert [span.kind for span in replay.generated_spans] == [
        "tool_name",
        "argument_value",
    ]
    assert replay.generated_spans[0].label == "tool[0]=get_reservation_details"
    assert replay.generated_spans[1].label == "tool[0].reservation_id"
    for span in replay.generated_spans:
        assert replay.response.token_ids[span.start : span.end] == span.token_ids
        assert len(span.prediction_positions) == len(span.token_ids)


def test_qwen35_style_template_handles_history_and_context_merged_tool_name():
    tokenizer = ContextMergingTokenizer()
    call = sample_call()
    call["request"]["messages"].append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_booking",
                        "arguments": '{"booking_id":"ABC123"}',
                    },
                }
            ],
        }
    )
    call["request"]["messages"].append(
        {"role": "tool", "content": '{"status":"ok"}'}
    )

    replay = render_actual_response(tokenizer, call, enable_thinking=False)
    candidate = build_tool_candidate(
        tokenizer, call, "cancel_reservation", enable_thinking=False
    )

    tool_span = next(span for span in replay.generated_spans if span.kind == "tool_name")
    assert "get_reservation_details" in tokenizer.decode(tool_span.token_ids)
    assert "cancel_reservation" in tokenizer.decode(candidate.name_token_ids)
    assert tuple(tokenizer.encode("cancel_reservation")) != candidate.name_token_ids


def test_infer_first_error_prefers_verified_schema_error_and_selects_history(tmp_path):
    valid = sample_call()
    valid["response"]["tool_calls"][0]["name"] = "cancel_reservation"
    valid["response"]["tool_calls"][0]["arguments"] = {}
    invalid = sample_call()
    invalid["call_id"] = "call-2"
    invalid["response"]["tool_calls"][0]["name"] = "invented_tool"
    case = Tau2Case(
        task_id="5",
        simulation_id="failed",
        reward=0.0,
        expected_tools=("cancel_reservation",),
        actual_tools=("cancel_reservation", "invented_tool"),
        task={},
        simulation={"reward_info": {"reward": 0.0}, "messages": []},
    )
    calls = [
        LoggedCall(tmp_path / "a.json", valid, call_index=0),
        LoggedCall(tmp_path / "b.json", invalid, call_index=1),
    ]

    event = infer_first_error(case, calls)

    assert event is not None
    assert event.call_index == 1
    assert event.kind == "tool_schema_error"
    assert event.confidence == "verified"
    assert select_event_calls(calls, event.call_index, before=None, after=0) == calls


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
