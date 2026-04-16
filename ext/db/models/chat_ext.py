"""Helpers for the `chats.selected_kb_config` JSONB column.

Shape (per workflow spec §3.2):
    [ {"kb_id": int, "subtag_ids": [int, ...]}, ... ]
    empty subtag_ids = "all subtags in this KB"
    None/missing      = "no KB selected; private docs only"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class SelectedKBConfig:
    kb_id: int
    subtag_ids: List[int] = field(default_factory=list)


def validate_selected_kb_config(raw: Any) -> Optional[List[SelectedKBConfig]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("selected_kb_config must be a list or null")

    result: List[SelectedKBConfig] = []
    seen_kb_ids: set[int] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {i} is not an object")
        if "kb_id" not in entry:
            raise ValueError(f"entry {i} missing kb_id")
        kb_id = entry["kb_id"]
        if not isinstance(kb_id, int) or isinstance(kb_id, bool):
            raise ValueError(f"entry {i} kb_id must be an int")
        if kb_id in seen_kb_ids:
            raise ValueError(f"entry {i} duplicate kb_id={kb_id}")
        seen_kb_ids.add(kb_id)

        subtag_ids = entry.get("subtag_ids", [])
        if not isinstance(subtag_ids, list):
            raise ValueError(f"entry {i} subtag_ids must be a list")
        for j, sid in enumerate(subtag_ids):
            if not isinstance(sid, int) or isinstance(sid, bool):
                raise ValueError(f"entry {i} subtag_ids[{j}] must be an int")
        if len(set(subtag_ids)) != len(subtag_ids):
            raise ValueError(f"entry {i} subtag_ids contains duplicates")

        result.append(SelectedKBConfig(kb_id=kb_id, subtag_ids=list(subtag_ids)))
    return result
