# src/rag_pipeline.py
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def build_rag_pipeline(documents: list[str]):
    """
    Builds a simple RAG pipeline from a list of text documents.
    Returns a callable chain.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db         = FAISS.from_texts(documents, embeddings)
    retriever  = db.as_retriever(search_kwargs={"k": 2})
    llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only the context below.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain