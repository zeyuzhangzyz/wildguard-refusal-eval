from wildguard_refusal_eval.judge import parse_output, select_shard


def test_parse_complete_official_output() -> None:
    parsed = parse_output("Harmful request: no\nResponse refusal: yes\nHarmful response: no\n")
    assert parsed == {"harmful_request": False, "response_refusal": True, "harmful_response": False}


def test_shard_filters_arm_ids() -> None:
    rows = [
        {"example_id": "a", "prompt": "p", "response": "r", "arm_id": 29},
        {"example_id": "b", "prompt": "p", "response": "r", "arm_id": 30},
        {"example_id": "c", "prompt": "p", "response": "r", "arm_id": 44},
        {"example_id": "d", "prompt": "p", "response": "r", "arm_id": 45},
    ]
    assert [row["example_id"] for row in select_shard(rows, 30, 45)] == ["b", "c"]
