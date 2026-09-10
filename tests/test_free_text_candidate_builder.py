from app.services.free_text_candidate_builder import (
    extract_free_text_candidate_proposals,
)


def test_builder_extracts_explicit_bilingual_candidate_types():
    proposals = extract_free_text_candidate_proposals(
        "I prefer explanations with concrete examples. "
        "My target role is Agent Engineer. "
        "Correction: my location is Shanghai. "
        "I learned that checkpoints need schema versions."
    )

    assert [proposal.category for proposal in proposals] == [
        "preference",
        "fact",
        "correction",
        "episode",
    ]
    assert proposals[1].memory_key == "user-target-role"
    assert all(proposal.confidence == 1.0 for proposal in proposals)


def test_builder_ignores_implicit_or_third_party_statements():
    proposals = extract_free_text_candidate_proposals(
        "Please explain token budgeting. We prefer PostgreSQL in production. "
        "The target role is unclear."
    )

    assert proposals == ()


def test_user_input_trace_creates_reviewable_journal_records(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Capture explicit user memory candidates",
            "user_goal": "I prefer explanations with concrete examples and tradeoffs.",
            "success_criteria": ["Candidate decisions are auditable."],
        },
    ).json()

    initial = client.get(f"/agent/memory-candidates?task_id={task['id']}").json()
    assert len(initial) == 1
    assert initial[0]["decision"] == "needs_review"
    assert initial[0]["storage_action"] == "not_stored"
    assert initial[0]["source"] == "user_input"
    assert initial[0]["provenance"]["candidate_category"] == "preference"
    assert client.get("/agent/memories").json() == []

    trace = client.post(
        f"/agent/tasks/{task['id']}/traces",
        json={
            "event_type": "user_input",
            "input_summary": "Correction: my target role is Platform Engineer.",
            "output_summary": "Captured a user correction.",
        },
    ).json()
    candidates = client.get(
        f"/agent/memory-candidates?task_id={task['id']}"
    ).json()

    assert trace["metadata_json"]["candidate_builder"]["candidate_count"] == 1
    assert len(candidates) == 2
    correction = candidates[0]
    assert correction["decision"] == "needs_review"
    assert correction["memory_key"] == "user-target-role"
    assert correction["provenance"]["candidate_category"] == "correction"
    assert correction["provenance"]["execution_trace_id"] == trace["id"]
    assert client.get("/agent/memories").json() == []
