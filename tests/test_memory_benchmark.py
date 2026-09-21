from app.memory_benchmark import run_memory_benchmark


def test_memory_benchmark_meets_current_quality_floor():
    report = run_memory_benchmark()

    assert report.candidate_case_count == 7
    assert report.candidate_precision >= 1.0
    assert report.candidate_recall >= 1.0
    assert report.decision_accuracy >= 1.0
    assert report.extraction_case_count == 11
    assert report.extraction_precision >= 1.0
    assert report.extraction_recall >= 1.0
    assert report.extraction_false_positive_rate == 0.0
    assert report.retrieval_recall >= 1.0
    assert report.lexical_semantic_retrieval_recall == 0.0
    assert report.hybrid_semantic_retrieval_recall == 1.0
    assert report.retrieval_case_count == 3
    assert report.lexical_retrieval_recall_at_1 == 1 / 3
    assert report.semantic_retrieval_recall_at_1 == 2 / 3
    assert report.hybrid_retrieval_recall_at_1 == 1.0
    assert report.routed_retrieval_recall_at_1 == 1.0
    assert report.routed_retrieval_embedding_calls == 2
    assert report.hybrid_retrieval_regression_count == 0
    assert [case.case_id for case in report.retrieval_cases] == [
        "exact-identifier",
        "semantic-paraphrase",
        "cross-language",
    ]
    assert [case.route_strategy for case in report.retrieval_cases] == [
        "lexical",
        "hybrid",
        "hybrid",
    ]
    assert report.retrieval_scope_leakage_count == 0
    assert report.pre_compaction_critical_constraint_retention >= 1.0
    assert report.post_compaction_critical_constraint_retention >= 1.0
