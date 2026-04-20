"""P2.3: private chat uploads route to the consolidated ``chat_private``
collection (not ``chat_{chat_id}``). Both chat_id and owner_user_id get
stamped in the payload so retrieval can tenant-filter correctly."""
from __future__ import annotations

import inspect

from ext.routers import upload
from ext.services.vector_store import CHAT_PRIVATE_COLLECTION


def test_upload_source_uses_chat_private_constant() -> None:
    """Handler references the CHAT_PRIVATE_COLLECTION constant, not a
    dynamic ``chat_{chat_id}`` string."""
    src = inspect.getsource(upload)
    # Old pattern must not appear as a write target anymore
    assert 'collection=f"chat_{chat_id}"' not in src
    # New pattern should be present
    assert "CHAT_PRIVATE_COLLECTION" in src


def test_payload_stamps_both_chat_id_and_owner() -> None:
    """The handler must build a payload_base carrying both chat_id and
    owner_user_id so the retrieval-side filter can find the user's docs."""
    src = inspect.getsource(upload)
    # Accept either dict-literal style or kwarg-style; just confirm both keys are present
    assert '"chat_id": chat_id' in src
    assert '"owner_user_id"' in src


def test_constant_imported_in_upload() -> None:
    """Sanity: the constant comes from vector_store."""
    assert upload.CHAT_PRIVATE_COLLECTION == CHAT_PRIVATE_COLLECTION
    assert CHAT_PRIVATE_COLLECTION == "chat_private"
