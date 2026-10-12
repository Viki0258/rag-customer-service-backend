"""LangChain adapters; domain policy and provider contracts remain authoritative.

Construct chains per request so scoped knowledge/prompt overrides cannot retain
another request's model scope or conversation. No external tracing is enabled.
"""
from dataclasses import asdict, replace
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langsmith import tracing_context

from whatsapp_support.domain import RetrievalQuery, RetrievedChunk


class RagStageError(Exception):
    """Stable public error code, without provider text or credentials."""

    def __init__(self, code: str):
        super().__init__(code)
        self.error_code = code


class HybridKnowledgeRetriever(BaseRetriever):
    """Async, request-scoped adapter over the existing embedding/knowledge ports."""

    embedding: Any
    knowledge: Any
    query_context: RetrievalQuery

    def _get_relevant_documents(self, query, *, run_manager):
        raise NotImplementedError("Use ainvoke for the asynchronous knowledge ports")

    async def _aget_relevant_documents(self, query, *, run_manager):
        try:
            vectors = await self.embedding.embed([query])
            if len(vectors) != 1 or not isinstance(vectors[0], list):
                raise ValueError("embedding response shape mismatch")
        except Exception as exc:
            raise RagStageError("embedding_failed") from exc
        try:
            chunks = await self.knowledge.retrieve(replace(
                self.query_context, query_text=query,
                query_vector=tuple(float(value) for value in vectors[0]),
                embedding_model=self.embedding.model_name,
                embedding_dimension=self.embedding.dimension,
            ))
            return [Document(id=chunk.chunk_id, page_content=chunk.text,
                             metadata={"retrieved_chunk": asdict(chunk)})
                    for chunk in chunks]
        except Exception as exc:
            raise RagStageError("knowledge_index_unavailable") from exc


def documents_to_chunks(documents):
    # Preserve IDs, ranks and metadata exactly, including untrusted source text.
    return [RetrievedChunk(**{**doc.metadata["retrieved_chunk"], "text": doc.page_content})
            for doc in documents]


def retrieval_chain(*, embedding, knowledge, query_context):
    retriever = HybridKnowledgeRetriever(
        embedding=embedding, knowledge=knowledge, query_context=query_context)
    return retriever | RunnableLambda(documents_to_chunks, name="restore_evidence_contract")


def answer_chain(*, prompt_builder, model, validator):
    """Prompt -> existing structured model adapter -> mandatory domain validation."""
    def build_prompt(state):
        return prompt_builder.build(**state["prompt_inputs"])

    async def generate(state):
        try:
            return await model.generate(state["request"])
        except Exception as exc:
            code = getattr(exc, "error_code", None)
            raise RagStageError(code if code in {"model_failed", "model_schema_invalid"}
                                else "model_failed") from exc

    def validate(state):
        inputs = state["prompt_inputs"]
        return validator.validate(state["model_result"], inputs["retrieved_chunks"],
                                  inputs["locale"], inputs["trace_id"], state["limits"])

    return (
        RunnablePassthrough.assign(request=RunnableLambda(build_prompt, name="grounded_prompt"))
        | RunnablePassthrough.assign(model_result=RunnableLambda(generate, name="structured_generation"))
        | RunnableLambda(validate, name="evidence_and_reply_validation")
    )


async def invoke_private(chain, value):
    # Do not upload conversation/evidence even if the host has tracing env vars.
    # Context-local (not a process-global environment mutation).
    with tracing_context(enabled=False):
        return await chain.ainvoke(value, config={"callbacks": []})
