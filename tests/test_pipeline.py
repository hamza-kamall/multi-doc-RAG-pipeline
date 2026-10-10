from types import SimpleNamespace

from langchain_core.documents import Document

from app.rag import pipeline


class StaticRunnable:
    def __init__(self, value):
        self.value = value

    def invoke(self, payload):
        return self.value(payload) if callable(self.value) else self.value


class FakeRetrieverMap:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, questions):
        self.owner.questions = questions
        return [[self.owner.document] for _ in questions]


class FakeRetriever:
    def __init__(self, document):
        self.document = document
        self.questions = []

    def map(self):
        return FakeRetrieverMap(self)


def tool_route(name: str):
    return SimpleNamespace(
        additional_kwargs={
            "tool_calls": [{"function": {"name": name}}]
        }
    )


def no_tool_route():
    return SimpleNamespace(additional_kwargs={})


def configure_successful_graders(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "retrieval_grader",
        StaticRunnable(SimpleNamespace(score="yes")),
    )
    monkeypatch.setattr(
        pipeline,
        "hallucination_grader",
        StaticRunnable(SimpleNamespace(hall_score="yes")),
    )
    monkeypatch.setattr(
        pipeline,
        "answer_grader",
        StaticRunnable(SimpleNamespace(score="yes")),
    )


def test_document_route_runs_multi_query_retrieval_and_generation(monkeypatch):
    document = Document(
        page_content="The document explains FastAPI.",
        metadata={"chunk_id": "doc-1-0", "document_id": "doc-1"},
    )
    retriever = FakeRetriever(document)
    monkeypatch.setattr(pipeline, "question_router", StaticRunnable(tool_route("vectorstoresearch")))
    monkeypatch.setattr(pipeline, "generate_alternate_queries", lambda question: ["alt one", "alt two"])
    monkeypatch.setattr(pipeline, "reranker", lambda query, docs: {"documents": docs})
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, context: "grounded answer")
    configure_successful_graders(monkeypatch)

    result = pipeline.app(retriever).invoke({
        "question": "What does the document explain?",
        "document_id": "doc-1",
        "retry_count": 0,
    })

    assert retriever.questions == [
        "What does the document explain?", "alt one", "alt two"
    ]
    assert result["generation"] == "grounded answer"
    assert result["retry_count"] == 1
    assert result["documents"][0].metadata["chunk_id"] == "doc-1-0"


def test_general_question_uses_direct_llm_fallback(monkeypatch):
    retriever = FakeRetriever(Document(page_content="unused"))
    monkeypatch.setattr(pipeline, "question_router", StaticRunnable(no_tool_route()))
    monkeypatch.setattr(
        pipeline,
        "generate_general_answer",
        lambda question: "Hello from the generation model",
    )

    result = pipeline.app(retriever).invoke({
        "question": "Hello",
        "document_id": "doc-1",
        "retry_count": 0,
    })

    assert result["generation"] == "Hello from the generation model"
    assert retriever.questions == []


def test_web_route_uses_tavily_results(monkeypatch):
    retriever = FakeRetriever(Document(page_content="unused"))
    monkeypatch.setattr(pipeline, "question_router", StaticRunnable(tool_route("web_search")))
    monkeypatch.setattr(pipeline.settings, "tavily_api_key", "test-key")
    monkeypatch.setattr(pipeline, "TavilySearchAPIWrapper", lambda **kwargs: object())

    class FakeTavilyTool:
        def __init__(self, **kwargs):
            pass

        def invoke(self, payload):
            return [{"content": "Current web information", "url": "https://example.com"}]

    monkeypatch.setattr(pipeline, "TavilySearchResults", FakeTavilyTool)
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, context: "web answer")
    configure_successful_graders(monkeypatch)

    result = pipeline.app(retriever).invoke({
        "question": "What happened today?",
        "document_id": "doc-1",
        "retry_count": 0,
    })

    assert result["generation"] == "web answer"
    assert result["documents"][0].metadata["source_type"] == "web"


def test_unsupported_generation_stops_after_three_attempts(monkeypatch):
    document = Document(
        page_content="Evidence",
        metadata={"chunk_id": "doc-1-0", "document_id": "doc-1"},
    )
    retriever = FakeRetriever(document)
    monkeypatch.setattr(pipeline, "question_router", StaticRunnable(tool_route("vectorstoresearch")))
    monkeypatch.setattr(pipeline, "generate_alternate_queries", lambda question: [])
    monkeypatch.setattr(pipeline, "reranker", lambda query, docs: {"documents": docs})
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, context: "unsupported")
    monkeypatch.setattr(
        pipeline,
        "retrieval_grader",
        StaticRunnable(SimpleNamespace(score="yes")),
    )
    monkeypatch.setattr(
        pipeline,
        "hallucination_grader",
        StaticRunnable(SimpleNamespace(hall_score="no")),
    )

    result = pipeline.app(retriever).invoke({
        "question": "Question",
        "document_id": "doc-1",
        "retry_count": 0,
    })

    assert result["retry_count"] == 3
    assert result["generation"] == "unsupported"
