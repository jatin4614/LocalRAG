"""Minimal ORM models mirroring the upstream-compatible tables our KB FKs point at.

These exist so our code and tests can read/write users/groups/chats without
booting upstream Open WebUI. In production, the same tables are populated by
upstream's auth layer; we only read them here (plus chat.selected_kb_config).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class User(Base):
    __tablename__ = "users"

    id:            Mapped[str] = mapped_column(String, primary_key=True)  # UUID string in upstream
    email:         Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role:          Mapped[str] = mapped_column(String, nullable=False, default="user")


class Group(Base):
    __tablename__ = "groups"

    id:   Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class UserGroup(Base):
    __tablename__ = "user_groups"

    # ``users.id`` is a UUID string in upstream; the FK side must match.
    user_id:  Mapped[str] = mapped_column(String, ForeignKey("users.id",  ondelete="CASCADE"), primary_key=True)
    group_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)


class Chat(Base):
    __tablename__ = "chats"

    id:                 Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # ``users.id`` is a UUID string in upstream; FK side must match.
    user_id:            Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title:              Mapped[Optional[str]] = mapped_column(String)
    created_at:         Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    selected_kb_config: Mapped[Optional[list]] = mapped_column(JSON)  # JSONB on PG, JSON on SQLite
