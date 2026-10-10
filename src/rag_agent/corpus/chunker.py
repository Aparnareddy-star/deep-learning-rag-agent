
"""
Document loading and chunking for the RAG workshop.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings


class DocumentChunker:
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Invalid chunk size or overlap")

        suffix = file_path.suffix.lower()

        if suffix == ".md":
            raw_chunks = self._chunk_markdown(
                file_path, chunk_size, chunk_overlap
            )
        elif suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path, chunk_size, chunk_overlap
            )
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}"
            )

        metadata = self._infer_metadata(
            file_path, metadata_overrides
        )

        chunks = []

        for index, raw in enumerate(raw_chunks):
            content = raw["text"].strip()

            if not content:
                continue

            chunk_id = hashlib.sha256(
                (
                    file_path.name
                    + ":"
                    + str(index)
                    + ":"
                    + content
                ).encode("utf-8")
            ).hexdigest()

            chunk_metadata = metadata.model_copy(
                update={
                    "page": raw.get("page"),
                    "section": raw.get("header", ""),
                }
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                metadata=chunk_metadata,
            )

            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from {file_path.name}"
        )

        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:

        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path,
                    metadata_overrides=metadata_overrides,
                )
                all_chunks.extend(chunks)

            except Exception as exc:
                logger.error(
                    f"Failed to process {file_path}: {exc}"
                )
                raise

        return all_chunks

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:

        from langchain_community.document_loaders import (
            PyPDFLoader,
        )

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        results = []

        for page in pages:
            parts = splitter.split_text(
                page.page_content
            )

            for part in parts:
                if part.strip():
                    results.append({
                        "text": part,
                        "page": page.metadata.get("page", 0) + 1,
                    })

        return results

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:

        text = file_path.read_text(
            encoding="utf-8"
        )

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        sections = header_splitter.split_text(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        results = []

        for section in sections:
            header = " / ".join(
                str(value)
                for value in section.metadata.values()
            )

            pieces = splitter.split_text(
                section.page_content
            )

            for piece in pieces:
                if piece.strip():
                    results.append({
                        "text": piece,
                        "header": header,
                    })

        return results

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:

        file_path = Path(file_path)
        stem = file_path.stem.lower()
        parts = stem.split("_")

        valid_difficulties = {
            "beginner",
            "intermediate",
            "advanced",
        }

        difficulty = "beginner"
        topic_parts = parts

        if parts[-1] in valid_difficulties:
            difficulty = parts[-1]
            topic_parts = parts[:-1]

        topic = " ".join(topic_parts).upper()

        bonus_topics = {
            "som",
            "boltzmannmachine",
            "gan",
        }

        normalized_topic = topic.lower().replace(" ", "")

        data = {
            "source": file_path.name,
            "topic": topic,
            "difficulty": difficulty,
            "is_bonus": normalized_topic in bonus_topics,
        }

        if overrides:
            data.update(overrides)

        return ChunkMetadata(**data)
