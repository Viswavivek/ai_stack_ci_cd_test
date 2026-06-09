# tests/test_ai_quality.py
import os
import json
import pytest

# skip entire file if no API key — avoids failures in local dev
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping AI quality tests"
)


# ════════════════════════════════════════════════════════════════
# SENTIMENT CLASSIFICATION QUALITY TESTS
# ════════════════════════════════════════════════════════════════

class TestSentimentQuality:
    """Tests that LLM classifies sentiment correctly."""

    # test cases: (text, expected_sentiment)
    CASES = [
        ("I love this product! It's amazing.",        "positive"),
        ("This is terrible. Worst experience ever.",  "negative"),
        ("The meeting is at 3pm tomorrow.",            "neutral"),
        ("Great service, very happy with results.",   "positive"),
        ("Completely broken, waste of money.",        "negative"),
    ]

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import here so module-level skip works correctly."""
        from src.llm_utils import classify_sentiment
        self.classify = classify_sentiment

    @pytest.mark.parametrize("text,expected", CASES)
    def test_sentiment_accuracy(self, text, expected):
        """LLM should classify each text with correct sentiment."""
        result = self.classify(text)
        assert result == expected, (
            f"Expected '{expected}' for '{text[:40]}...' "
            f"but got '{result}'"
        )

    def test_sentiment_accuracy_rate(self):
        """At least 80% of test cases should be classified correctly."""
        from src.llm_utils import classify_sentiment
        correct = 0
        for text, expected in self.CASES:
            result = classify_sentiment(text)
            if result == expected:
                correct += 1

        accuracy = correct / len(self.CASES)
        assert accuracy >= 0.80, (
            f"Sentiment accuracy {accuracy:.0%} below 80% threshold"
        )


# ════════════════════════════════════════════════════════════════
# SUMMARIZATION QUALITY TESTS
# ════════════════════════════════════════════════════════════════

class TestSummarizationQuality:
    """Tests that summaries are concise and relevant."""

    TEXT = (
        "Python is a high-level, interpreted programming language known for "
        "its clear syntax and readability. It was created by Guido van Rossum "
        "and first released in 1991. Python supports multiple programming "
        "paradigms including procedural, object-oriented, and functional "
        "programming. It is widely used in data science, web development, "
        "automation, and artificial intelligence."
    )

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.llm_utils import summarize_text
        self.summarize = summarize_text

    def test_summary_under_word_limit(self):
        """Summary should not exceed max_words."""
        summary   = self.summarize(self.TEXT, max_words=20)
        word_count = len(summary.split())
        assert word_count <= 25, (   # allow 25% tolerance
            f"Summary too long: {word_count} words (limit 20)"
        )

    def test_summary_not_empty(self):
        """Summary should not be empty."""
        summary = self.summarize(self.TEXT, max_words=50)
        assert len(summary.strip()) > 0

    def test_summary_mentions_python(self):
        """Summary should mention the main topic."""
        summary = self.summarize(self.TEXT, max_words=50)
        assert "python" in summary.lower(), (
            "Summary should mention Python — main topic of the text"
        )


# ════════════════════════════════════════════════════════════════
# RAG QUALITY TESTS
# ════════════════════════════════════════════════════════════════

class TestRAGQuality:
    """Tests that RAG pipeline retrieves and answers correctly."""

    DOCUMENTS = [
        "LangChain is a framework for building LLM applications.",
        "FAISS is a vector database by Meta for similarity search.",
        "RAG stands for Retrieval Augmented Generation.",
        "LangSmith is used for tracing and debugging LLM applications.",
        "Python is the most popular language for AI development.",
    ]

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.rag_pipeline import build_rag_pipeline
        self.chain = build_rag_pipeline(self.DOCUMENTS)

    def test_rag_answers_correctly(self):
        """RAG should answer basic factual question from documents."""
        answer = self.chain.invoke("What is LangChain?")
        assert "langchain" in answer.lower() or "llm" in answer.lower(), (
            f"RAG answer doesn't mention LangChain or LLM: {answer}"
        )

    def test_rag_answers_faiss_question(self):
        """RAG should correctly answer question about FAISS."""
        answer = self.chain.invoke("What is FAISS?")
        assert "meta" in answer.lower() or "vector" in answer.lower() \
               or "similarity" in answer.lower(), (
            f"RAG answer about FAISS seems wrong: {answer}"
        )

    def test_rag_not_empty(self):
        """RAG should always return a non-empty answer."""
        answer = self.chain.invoke("What is RAG?")
        assert len(answer.strip()) > 0


# ════════════════════════════════════════════════════════════════
# QUALITY GATE — overall pass/fail for deployment
# ════════════════════════════════════════════════════════════════

class TestQualityGate:
    """
    Final quality gate — all these must pass before deployment.
    If any fail the pipeline blocks deployment.
    """

    def test_sentiment_gate(self):
        """Sentiment classifier must achieve 80%+ accuracy."""
        from src.llm_utils import classify_sentiment
        cases = [
            ("I love this!", "positive"),
            ("This is awful", "negative"),
            ("The sky is blue", "neutral"),
        ]
        correct = sum(
            1 for text, expected in cases
            if classify_sentiment(text) == expected
        )
        accuracy = correct / len(cases)
        assert accuracy >= 0.80, f"Quality gate FAILED: {accuracy:.0%} accuracy"

    def test_answer_quality_gate(self):
        """QA system must answer basic questions correctly."""
        from src.llm_utils import answer_question
        answer = answer_question(
            context  = "The capital of India is New Delhi.",
            question = "What is the capital of India?"
        )
        assert "new delhi" in answer.lower() or "delhi" in answer.lower(), (
            f"Quality gate FAILED: wrong answer — {answer}"
        )