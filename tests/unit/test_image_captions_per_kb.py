"""Per-KB image_captions schema + helper tests.

Mirrors the should_contextualize precedence: per-KB explicit value
wins (True OR False — both are explicit operator decisions), falling
back to the global ``RAG_IMAGE_CAPTIONS`` env flag.
"""
from ext.services import kb_config
from ext.services.ingest import should_caption_images


# --- Schema additions ---


class TestImageCaptionsValidate:
    def test_true_kept(self) -> None:
        assert kb_config.validate_config(
            {"image_captions": True}
        ) == {"image_captions": True}

    def test_false_kept(self) -> None:
        assert kb_config.validate_config(
            {"image_captions": False}
        ) == {"image_captions": False}

    def test_string_truthy_coerced(self) -> None:
        assert kb_config.validate_config(
            {"image_captions": "1"}
        ) == {"image_captions": True}

    def test_string_falsy_coerced(self) -> None:
        assert kb_config.validate_config(
            {"image_captions": "0"}
        ) == {"image_captions": False}


class TestImageCaptionsMerge:
    def test_or_wins_across_kbs(self) -> None:
        merged = kb_config.merge_configs([
            {"image_captions": False},
            {"image_captions": True},
        ])
        assert merged == {"image_captions": True}


class TestImageCaptionsOverlay:
    def test_excluded_from_env_overlay(self) -> None:
        """image_captions is INGEST_ONLY — must NOT leak into the
        request-time env overlay (it has no meaning at request time)."""
        env = kb_config.config_to_env_overrides({"image_captions": True})
        assert "RAG_IMAGE_CAPTIONS" not in env
        assert env == {}


# --- should_caption_images precedence ---


class TestShouldCaptionImages:
    def test_per_kb_true_wins_over_env_off(self) -> None:
        assert should_caption_images(
            env_flag="0", kb_rag_config={"image_captions": True},
        ) is True

    def test_per_kb_false_wins_over_env_on(self) -> None:
        # The point of the helper: a per-KB OFF must override env=ON.
        assert should_caption_images(
            env_flag="1", kb_rag_config={"image_captions": False},
        ) is False

    def test_no_per_kb_falls_back_to_env_on(self) -> None:
        assert should_caption_images(
            env_flag="1", kb_rag_config={},
        ) is True

    def test_no_per_kb_falls_back_to_env_off(self) -> None:
        assert should_caption_images(
            env_flag="0", kb_rag_config={},
        ) is False

    def test_env_none_treated_as_off(self) -> None:
        assert should_caption_images(
            env_flag=None, kb_rag_config=None,
        ) is False

    def test_env_string_only_1_enables(self) -> None:
        # Mirrors RAG_* convention: only literal "1" means on.
        for val in ["", "0", "true", "yes", "TRUE"]:
            assert should_caption_images(
                env_flag=val, kb_rag_config=None,
            ) is False
