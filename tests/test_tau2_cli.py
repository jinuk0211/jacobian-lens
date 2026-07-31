import json

from scripts.analyze_tau2 import build_manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_manifest_reports_missing_and_selected_logs(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "results.json",
        {
            "tasks": [
                {
                    "id": "1",
                    "evaluation_criteria": {
                        "actions": [{"name": "cancel_reservation"}]
                    },
                },
                {"id": "2", "evaluation_criteria": {"actions": []}},
            ],
            "simulations": [
                {
                    "id": "sim-1",
                    "task_id": "1",
                    "reward_info": {"reward": 0.0},
                    "messages": [],
                },
                {
                    "id": "sim-2",
                    "task_id": "2",
                    "reward_info": {"reward": 0.0},
                    "messages": [],
                },
            ],
        },
    )
    for call_id, tool_name in (("a", "lookup_booking"), ("b", "search_flights")):
        write_json(
            run_dir
            / "artifacts"
            / "task_1"
            / "sim_sim-1"
            / "llm_debug"
            / f"{call_id}.json",
            {
                "call_id": call_id,
                "call_name": "agent_response",
                "request": {"messages": [], "tools": []},
                "response": {"tool_calls": [{"name": tool_name}]},
            },
        )

    manifest, selected = build_manifest(run_dir, call_selection="last")

    assert manifest["summary"] == {
        "saved_cases": 2,
        "selected_cases": 2,
        "selected_calls": 1,
        "cases_without_agent_logs": 1,
    }
    assert manifest["cases"][0]["call_ids"] == ["b"]
    assert manifest["cases"][0]["candidate_tools"] == [
        "cancel_reservation",
        "search_flights",
    ]
    assert manifest["cases"][1]["status"] == "missing_agent_logs"
    assert len(selected) == 1
