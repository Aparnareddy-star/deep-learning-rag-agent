
"""
Central configuration and model factories for the RAG application.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class EmbeddingProvider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Language model settings
    llm_provider: LLMProvider = LLMProvider.GROQ

    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
    )

    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        alias="GROQ_MODEL",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )

    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL",
    )

    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        alias="LMSTUDIO_BASE_URL",
    )

    lmstudio_model: str = Field(
        default="local-model",
        alias="LMSTUDIO_MODEL",
    )

    # Embedding settings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # ChromaDB settings
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_DB_PATH",
    )

    chroma_collection_name: str = Field(
        default="deep_learning_corpus",
        alias="CHROMA_COLLECTION_NAME",
    )

    # Retrieval settings
    retrieval_k: int = Field(
        default=4,
        alias="RETRIEVAL_K",
    )

    similarity_threshold: float = Field(
        default=0.3,
        alias="SIMILARITY_THRESHOLD",
    )

    max_context_tokens: int = Field(
        default=3000,
        alias="MAX_CONTEXT_TOKENS",
    )

    # Application settings
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    app_title: str = Field(
        default="Deep Learning Interview Prep Agent",
        alias="APP_TITLE",
    )

    corpus_dir: str = Field(
        default="./data/corpus",
        alias="CORPUS_DIR",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()


class LLMFactory:
    """Create the configured language model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self) -> BaseChatModel:
        provider = self._settings.llm_provider

        if provider == LLMProvider.GROQ:
            return self._create_groq()

        if provider == LLMProvider.OLLAMA:
            return self._create_ollama()

        if provider == LLMProvider.LMSTUDIO:
            return self._create_lmstudio()

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )

    def _create_groq(self) -> BaseChatModel:
        """Create the Groq language model."""

        from langchain_groq import ChatGroq

        api_key = self._settings.groq_api_key.strip()

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is missing. "
                "Add your API key to the .env file."
            )

        return ChatGroq(
            model=self._settings.groq_model,
            api_key=api_key,
            temperature=0,
            max_retries=2,
        )

    def _create_ollama(self) -> BaseChatModel:
        """Create an Ollama local language model."""

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self._settings.ollama_model,
            base_url=self._settings.ollama_base_url,
            temperature=0,
        )

    def _create_lmstudio(self) -> BaseChatModel:
        """Create an LM Studio language model."""

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self._settings.lmstudio_model,
            base_url=self._settings.lmstudio_base_url,
            api_key="lm-studio",
            temperature=0,
        )


class EmbeddingFactory:
    """Create the configured embedding model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self) -> Embeddings:
        provider = self._settings.embedding_provider

        if provider == EmbeddingProvider.LOCAL:
            return self._create_local()

        if provider == EmbeddingProvider.OPENAI:
            return self._create_openai()

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    def _create_local(self) -> Embeddings:
        """Create local Hugging Face embeddings."""

        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,
        )

        return HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    def _create_openai(self) -> Embeddings:
        """Create OpenAI embeddings."""

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
        )
