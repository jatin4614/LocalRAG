from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ..base import Base

# ``rag_config`` wants a JSONB column on Postgres but the test suite runs
# against SQLite where JSONB is unavailable. Using the dialect-agnostic
# ``JSON`` type keeps unit tests green while still mapping to JSONB in
# production via the server-side migration (006).
_RagConfigType = JSON().with_variant(JSONB, "postgresql")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    admin_id: Mapped[str] = mapped_column(String(255), nullable=False)  # UUID from upstream "user" table
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # P3.0: per-KB retrieval quality overrides. Empty dict = inherit
    # process-level defaults; populated by admin via PATCH /api/kb/{id}/config.
    # Merged UNION/MAX across selected KBs by ``ext.services.kb_config``.
    rag_config: Mapped[dict[str, Any]] = mapped_column(
        _RagConfigType, server_default="{}", default=dict, nullable=False,
    )

    subtags: Mapped[list["KBSubtag"]] = relationship(
        back_populates="kb", cascade="all, delete-orphan"
    )
    documents: Mapped[list["KBDocument"]] = relationship(
        back_populates="kb", cascade="all, delete-orphan"
    )
    access: Mapped[list["KBAccess"]] = relationship(
        back_populates="kb", cascade="all, delete-orphan"
    )


class KBSubtag(Base):
    __tablename__ = "kb_subtags"
    __table_args__ = (CheckConstraint("length(name) > 0", name="subtag_name_nonempty"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kb_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    kb: Mapped[KnowledgeBase] = relationship(back_populates="subtags")


class KBDocument(Base):
    __tablename__ = "kb_documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kb_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    subtag_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("kb_subtags.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    bytes: Mapped[Optional[int]] = mapped_column(BigInteger)  # TODO: expose in admin UI as file size
    ingest_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    uploaded_by: Mapped[str] = mapped_column(String(255), nullable=False)  # UUID
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)
    # P0.4: composite pipeline version stamped at ingest time. NULL for rows
    # inserted before the column existed. See ext/services/pipeline_version.py.
    pipeline_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # P2.8: sha256 of the original upload bytes in the BlobStore. NULL for
    # legacy rows and for sync-ingest uploads (no blob persisted). Used by
    # ext/services/blob_gc.py to free the blob after soft-delete retention.
    blob_sha: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    kb: Mapped[KnowledgeBase] = relationship(back_populates="documents")


class KBAccess(Base):
    __tablename__ = "kb_access"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kb_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(255))  # UUID
    group_id: Mapped[Optional[str]] = mapped_column(Text)
    access_type: Mapped[str] = mapped_column(String(20), default="read", nullable=False)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    kb: Mapped[KnowledgeBase] = relationship(back_populates="access")

    def __init__(self, **kwargs):
        u = kwargs.get("user_id")
        g = kwargs.get("group_id")
        if (u is None) == (g is None):
            raise ValueError("KBAccess requires exactly one of user_id or group_id")
        super().__init__(**kwargs)
