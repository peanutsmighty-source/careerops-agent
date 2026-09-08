from app.memory_benchmark import run_memory_benchmark


def test_memory_benchmark_meets_current_quality_floor():
    report = run_memory_benchmark()

    assert report.candidate_case_count == 7
    assert report.candidate_precision >= 1.0
    assert report.candidate_recall >= 1.0
    assert report.decision_accuracy >= 1.0
    assert report.retrieval_recall >= 1.0
    assert report.pre_compaction_critical_constraint_retention >= 1.0
