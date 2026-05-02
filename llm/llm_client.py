from __future__ import annotations

import os
from typing import Iterable, List, Optional

from dotenv import load_dotenv

# LangChain (core + OpenAI integrations)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class LLMClient:
    """
    Thin wrapper around LangChain + OpenAI for:
      - Chat completions via ChatOpenAI
      - Text embeddings via OpenAIEmbeddings

    Configuration:
      - OPENAI_API_KEY           (required)
      - OPENAI_MODEL             (default: gpt-5-mini)
      - OPENAI_EMBEDDINGS_MODEL  (default: text-embedding-3-small)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        embeddings_model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )

        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        emb_model = embeddings_model or os.getenv(
            "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"
        )

        # Chat LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
        )

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=emb_model,
            api_key=key,
        )

    def generate(self, prompt: str, system: Optional[str] = None, context: Optional[List] = None) -> str:
        """
        Simple single-turn chat generation with optional system instruction.
        """
        # Collapse context into a string (supports List[str] or LangChain Documents)
        if context is None:
            context_str = ""
        elif isinstance(context, str):
            context_str = context
        else:
            try:
                context_str = "\n\n".join(
                    [c.page_content if hasattr(c, "page_content") else str(c) for c in context]
                )
            except Exception:
                context_str = "\n\n".join([str(c) for c in context])

        system_msg = system or "You are a personal assistant who is researching on the personality and life of a person with access to their notes. Always answer in no less than 150 words."

        prompt_tmpl = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_msg}"),
                (
                    "human",
                    (
                        "Use the following pieces of retrieved context to answer the user's question. "
                        "If you don't know the answer, just say that you don't know.\n\n"
                        "Context:\n{context}\n\n"
                        "Question:\n{prompt}\n\n"
                        "Answer:"
                    ),
                ),
            ]
        )

        chain = prompt_tmpl | self.llm | StrOutputParser()
        return chain.invoke({"system_msg": system_msg, "context": context_str, "prompt": prompt})

    def embed_query(self, text: str) -> List[float]:
        """
        Return a single embedding vector for the query text.
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Return embeddings for a list of documents.
        """
        return self.embeddings.embed_documents(list(texts))
