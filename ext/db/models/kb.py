from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base


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
    bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    ingest_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    uploaded_by: Mapped[str] = mapped_column(String(255), nullable=False)  # UUID
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)

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
