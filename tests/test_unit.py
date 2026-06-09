# tests/test_unit.py
import pytest
from unittest.mock import patch, MagicMock


def _mock_chain(return_value):
    """Build a mock that survives two pipe (|) operations and returns return_value on invoke."""
    chain = MagicMock()
    chain.invoke.return_value = return_value

    mid = MagicMock()
    mid.__or__ = MagicMock(return_value=chain)

    prompt = MagicMock()
    prompt.__or__ = MagicMock(return_value=mid)
    return prompt, chain


# ── Test llm_utils ────────────────────────────────────────────
class TestClassifySentiment:

    def test_raises_on_empty_input(self):
        from src.llm_utils import classify_sentiment

        with pytest.raises(ValueError):
            classify_sentiment("")

    def test_raises_on_whitespace(self):
        from src.llm_utils import classify_sentiment

        with pytest.raises(ValueError):
            classify_sentiment("   ")

    @patch("src.llm_utils.StrOutputParser")
    @patch("src.llm_utils.ChatPromptTemplate")
    @patch("src.llm_utils.get_llm")
    def test_returns_lowercase(self, mock_get_llm, mock_prompt_cls, mock_parser_cls):
        from src.llm_utils import classify_sentiment

        mock_prompt, _ = _mock_chain("POSITIVE")
        mock_prompt_cls.from_template.return_value = mock_prompt

        result = classify_sentiment("I love this!")
        assert result == "positive"

    @patch("src.llm_utils.StrOutputParser")
    @patch("src.llm_utils.ChatPromptTemplate")
    @patch("src.llm_utils.get_llm")
    def test_valid_sentiments(self, mock_get_llm, mock_prompt_cls, mock_parser_cls):
        from src.llm_utils import classify_sentiment

        for label in ("positive", "negative", "neutral"):
            mock_prompt, _ = _mock_chain(label.upper())
            mock_prompt_cls.from_template.return_value = mock_prompt
            result = classify_sentiment("some text")
            assert result in ("positive", "negative", "neutral")


class TestSummarizeText:

    def test_raises_on_empty_input(self):
        from src.llm_utils import summarize_text

        with pytest.raises(ValueError):
            summarize_text("")

    def test_raises_on_empty_context(self):
        from src.llm_utils import answer_question

        with pytest.raises(ValueError):
            answer_question("", "What is this?")

    def test_raises_on_empty_question(self):
        from src.llm_utils import answer_question

        with pytest.raises(ValueError):
            answer_question("Some context", "")

    @patch("src.llm_utils.StrOutputParser")
    @patch("src.llm_utils.ChatPromptTemplate")
    @patch("src.llm_utils.get_llm")
    def test_summarize_returns_string(
        self, mock_get_llm, mock_prompt_cls, mock_parser_cls
    ):
        from src.llm_utils import summarize_text

        mock_prompt, _ = _mock_chain("A short summary.")
        mock_prompt_cls.from_template.return_value = mock_prompt

        result = summarize_text("Long text here.", max_words=10)
        assert result == "A short summary."

    @patch("src.llm_utils.StrOutputParser")
    @patch("src.llm_utils.ChatPromptTemplate")
    @patch("src.llm_utils.get_llm")
    def test_answer_question_returns_string(
        self, mock_get_llm, mock_prompt_cls, mock_parser_cls
    ):
        from src.llm_utils import answer_question

        mock_prompt, _ = _mock_chain("New Delhi.")
        mock_prompt_cls.from_template.return_value = mock_prompt

        result = answer_question(
            "Capital of India is New Delhi.", "What is the capital?"
        )
        assert result == "New Delhi."


class TestGetLLM:

    @patch("src.llm_utils.ChatOpenAI")
    def test_default_model(self, mock_openai):
        from src.llm_utils import get_llm

        get_llm()
        mock_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)

    @patch("src.llm_utils.ChatOpenAI")
    def test_custom_model(self, mock_openai):
        from src.llm_utils import get_llm

        get_llm(model="gpt-4o", temperature=0.5)
        mock_openai.assert_called_once_with(model="gpt-4o", temperature=0.5)


# ── Test rag_pipeline ─────────────────────────────────────────
class TestBuildRagPipeline:

    @patch("src.rag_pipeline.ChatOpenAI")
    @patch("src.rag_pipeline.FAISS")
    @patch("src.rag_pipeline.OpenAIEmbeddings")
    def test_returns_callable_chain(
        self, mock_embeddings_cls, mock_faiss_cls, mock_llm_cls
    ):
        from src.rag_pipeline import build_rag_pipeline

        mock_retriever = MagicMock()
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever
        mock_faiss_cls.from_texts.return_value = mock_db

        docs = ["LangChain is a framework.", "FAISS is a vector store."]
        chain = build_rag_pipeline(docs)

        mock_faiss_cls.from_texts.assert_called_once()
        mock_db.as_retriever.assert_called_once_with(search_kwargs={"k": 2})
        assert chain is not None

    @patch("src.rag_pipeline.ChatOpenAI")
    @patch("src.rag_pipeline.FAISS")
    @patch("src.rag_pipeline.OpenAIEmbeddings")
    def test_embeddings_model(self, mock_embeddings_cls, mock_faiss_cls, mock_llm_cls):
        from src.rag_pipeline import build_rag_pipeline

        mock_db = MagicMock()
        mock_db.as_retriever.return_value = MagicMock()
        mock_faiss_cls.from_texts.return_value = mock_db

        build_rag_pipeline(["some document"])
        mock_embeddings_cls.assert_called_once_with(model="text-embedding-3-small")

    @patch("src.rag_pipeline.ChatOpenAI")
    @patch("src.rag_pipeline.FAISS")
    @patch("src.rag_pipeline.OpenAIEmbeddings")
    def test_llm_model(self, mock_embeddings_cls, mock_faiss_cls, mock_llm_cls):
        from src.rag_pipeline import build_rag_pipeline

        mock_db = MagicMock()
        mock_db.as_retriever.return_value = MagicMock()
        mock_faiss_cls.from_texts.return_value = mock_db

        build_rag_pipeline(["some document"])
        mock_llm_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0)
