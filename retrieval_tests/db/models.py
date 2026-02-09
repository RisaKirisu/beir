import contextlib
from typing import AsyncIterator

from sqlalchemy import Index, Text, Computed, Integer, text, UUID
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


def get_async_engine(db_url: str):
    return create_async_engine(db_url, echo=False)


def get_session_maker(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@contextlib.asynccontextmanager
async def get_session(session_maker) -> AsyncIterator[AsyncSession]:
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def setup_pgvector(engine):
    """Enable pgvector extension and configure HNSW settings"""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("SET hnsw.iterative_scan = relaxed_order"))


# FTS Test Models - One table per test case (F0-F5)


class TestFTSChunkF0(Base):
    """F0: Baseline - chunk_text only, unweighted"""

    __tablename__ = "test_fts_chunk_f0"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    ts_vector = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk_text)", persisted=True),
    )

    __table_args__ = (
        Index("idx_test_fts_f0_ts_vector", "ts_vector", postgresql_using="gin"),
    )


class TestFTSChunkF1(Base):
    """F1: chunk_text + contextual_anchor, equal weight"""

    __tablename__ = "test_fts_chunk_f1"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_anchor: Mapped[str] = mapped_column(Text, nullable=False)

    ts_vector = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', chunk_text || ' ' || contextual_anchor)",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index("idx_test_fts_f1_ts_vector", "ts_vector", postgresql_using="gin"),
    )


class TestFTSChunkF2(Base):
    """F2: chunk_text + generated_questions, equal weight"""

    __tablename__ = "test_fts_chunk_f2"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    generated_questions: Mapped[str] = mapped_column(Text, nullable=False)

    ts_vector = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', chunk_text || ' ' || generated_questions)",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index("idx_test_fts_f2_ts_vector", "ts_vector", postgresql_using="gin"),
    )


class TestFTSChunkF3(Base):
    """F3: chunk_text + keywords_phrases, equal weight"""

    __tablename__ = "test_fts_chunk_f3"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    keywords_phrases: Mapped[str] = mapped_column(Text, nullable=False)

    ts_vector = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', chunk_text || ' ' || keywords_phrases)",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index("idx_test_fts_f3_ts_vector", "ts_vector", postgresql_using="gin"),
    )


class TestFTSChunkF4(Base):
    """F4: Full FTS with weighted fields (Keywords:A, Chunk:B, Questions:C, Anchor:D)"""

    __tablename__ = "test_fts_chunk_f4"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_anchor: Mapped[str] = mapped_column(Text, nullable=False)
    keywords_phrases: Mapped[str] = mapped_column(Text, nullable=False)
    generated_questions: Mapped[str] = mapped_column(Text, nullable=False)

    ts_vector = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', keywords_phrases), 'A') || "
            "setweight(to_tsvector('english', chunk_text), 'B') || "
            "setweight(to_tsvector('english', generated_questions), 'C') || "
            "setweight(to_tsvector('english', contextual_anchor), 'D')",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index("idx_test_fts_f4_ts_vector", "ts_vector", postgresql_using="gin"),
    )


# F5 uses same table as F4 (reranker is applied post-retrieval, not in DB)


# Vector Search Test Model


class TestVectorChunk(Base):
    """Model for vector search ablation tests (V0-V5)"""

    __tablename__ = "test_vector_chunk"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    retrieval_chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    interaction_id: Mapped[str] = mapped_column(UUID, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
