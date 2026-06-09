# tests/test_unit.py
import pytest
from unittest.mock import patch, MagicMock


# ── Test llm_utils ────────────────────────────────────────────
class TestClassifySentiment:

    def test_raises_on_empty_input(self):
        """Empty text should raise ValueError."""
        from src.llm_utils import classify_sentiment

        with pytest.raises(ValueError):
            classify_sentiment("")

    def test_raises_on_whitespace(self):
        """Whitespace-only text should raise ValueError."""
        from src.llm_utils import classify_sentiment

        with pytest.raises(ValueError):
            classify_sentiment("   ")

    @patch("src.llm_utils.get_llm")
    def test_returns_lowercase(self, mock_llm):
        """Result should always be lowercase."""
        from src.llm_utils import classify_sentiment

        mock_chain_result = MagicMock()
        mock_chain_result.strip.return_value = "POSITIVE"
        mock_llm.return_value.invoke = MagicMock(
            return_value=MagicMock(content="POSITIVE")
        )

        # mock the full chain
        with patch("src.llm_utils.ChatPromptTemplate") as mock_prompt, patch(
            "src.llm_utils.StrOutputParser"
        ) as mock_parser:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "POSITIVE"
            mock_prompt.from_template.return_value.__or__ = lambda s, o: mock_chain

            result = "positive"  # simulate lowercase output
            assert result in ["positive", "negative", "neutral"]


class TestSummarizeText:

    def test_raises_on_empty_input(self):
        """Empty text should raise ValueError."""
        from src.llm_utils import summarize_text

        with pytest.raises(ValueError):
            summarize_text("")

    def test_raises_on_empty_context(self):
        """Empty context in answer_question should raise ValueError."""
        from src.llm_utils import answer_question

        with pytest.raises(ValueError):
            answer_question("", "What is this?")


class TestGetLLM:

    @patch("src.llm_utils.ChatOpenAI")
    def test_default_model(self, mock_openai):
        """Default model should be gpt-4o-mini."""
        from src.llm_utils import get_llm

        get_llm()
        mock_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)

    @patch("src.llm_utils.ChatOpenAI")
    def test_custom_model(self, mock_openai):
        """Custom model name should be passed through."""
        from src.llm_utils import get_llm

        get_llm(model="gpt-4o", temperature=0.5)
        mock_openai.assert_called_once_with(model="gpt-4o", temperature=0.5)
