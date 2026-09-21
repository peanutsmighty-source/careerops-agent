from app.services.hybrid_retrieval import RetrievalDocument, rank_hybrid_documents
from app.services.memory_similarity import EmbeddingBatch
from app.services.retrieval_router import choose_retrieval_route


class CountingEmbeddingProvider:
    provider_version = "router-test-v1"

    def __init__(self):
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        return EmbeddingBatch(
            vectors=[[1.0, 0.0] for _ in texts],
            provider_version=self.provider_version,
            usage={"embedding_calls": 1, "embedding_tokens": len(texts)},
        )


DOCUMENTS = (
    RetrievalDocument(
        key="toolcall:1842",
        source_type="trace",
        content="ToolCall 1842 timeout",
        payload={},
    ),
    RetrievalDocument(
        key="toolcall:991",
        source_type="trace",
        content="External operation outcome unknown",
        payload={},
    ),
)


def test_short_exact_identifier_query_routes_to_lexical_without_embedding():
    provider = CountingEmbeddingProvider()
    query = "Inspect ToolCall 1842 timeout"

    route = choose_retrieval_route(query, embedding_available=True)
    result = rank_hybrid_documents(
        query,
        DOCUMENTS,
        embedding_provider=provider if route.strategy == "hybrid" else None,
        limit=1,
    )

    assert route.strategy == "lexical"
    assert route.reason == "short_exact_identifier_lookup"
    assert route.signals == ("ToolCall 1842",)
    assert result.candidates[0].document.key == "toolcall:1842"
    assert provider.call_count == 0


def test_long_or_semantic_query_keeps_the_default_hybrid_route():
    route = choose_retrieval_route(
        "Explain how a worker resumes after process interruption safely",
        embedding_available=True,
    )

    assert route.strategy == "hybrid"
    assert route.reason == "default_recall_route"


def test_missing_embedding_provider_routes_to_lexical_with_reason():
    route = choose_retrieval_route(
        "Recover work after process interruption",
        embedding_available=False,
    )

    assert route.strategy == "lexical"
    assert route.reason == "embedding_provider_unavailable"
