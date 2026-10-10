from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app.rag import generator


def test_generation_uses_separate_groq_model(monkeypatch):
    settings = SimpleNamespace(
        groq_generation_model="openai/gpt-oss-120b",
        groq_model="openai/gpt-oss-20b",
        groq_api_key="test-key",
    )
    monkeypatch.setattr(generator, "get_settings", lambda: settings)
    captured = {}

    def fake_chat_groq(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(generator, "ChatGroq", fake_chat_groq)
    generator.get_llm()
    assert captured == {
        "model": "openai/gpt-oss-120b",
        "api_key": "test-key",
        "temperature": 0,
    }


def test_document_generation_preserves_context_and_returns_text(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            assert "using only the context" in prompt
            assert "Example document facts" in prompt
            assert "What is an LLM?" in prompt
            return AIMessage(content="A language model.")

    monkeypatch.setattr(generator, "get_llm", lambda: FakeLLM())
    assert generator.generate_answer("What is an LLM?", "Example document facts") == "A language model."


def test_general_generation_returns_text(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            assert "Hello" in prompt
            return AIMessage(content="Hello back!")

    monkeypatch.setattr(generator, "get_llm", lambda: FakeLLM())
    assert generator.generate_general_answer("Hello") == "Hello back!"
