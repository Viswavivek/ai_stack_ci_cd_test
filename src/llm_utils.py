# src/llm_utils.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    """Returns configured LLM instance."""
    return ChatOpenAI(model=model, temperature=temperature)

def classify_sentiment(text: str) -> str:
    """
    Classifies text sentiment as positive, negative, or neutral.
    Returns one of: 'positive', 'negative', 'neutral'
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    llm    = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Classify the sentiment of this text as exactly one word: "
        "positive, negative, or neutral.\n\nText: {text}\n\nSentiment:"
    )
    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": text})
    return result.strip().lower()

def summarize_text(text: str, max_words: int = 50) -> str:
    """Summarizes text in max_words words."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    llm    = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in at most {max_words} words:\n\n"
        "{text}\n\nSummary:"
    )
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text, "max_words": max_words}).strip()

def answer_question(context: str, question: str) -> str:
    """Answers a question based on given context."""
    if not context or not question:
        raise ValueError("Context and question cannot be empty")

    llm    = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()