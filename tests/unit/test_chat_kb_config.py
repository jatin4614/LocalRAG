import pytest
from ext.db.models.chat_ext import validate_selected_kb_config, SelectedKBConfig

def test_none_is_valid():
    assert validate_selected_kb_config(None) is None

def test_valid_shape():
    cfg = [{"kb_id": 5, "subtag_ids": [12, 13]}, {"kb_id": 7, "subtag_ids": []}]
    got = validate_selected_kb_config(cfg)
    assert isinstance(got, list)
    assert all(isinstance(item, SelectedKBConfig) for item in got)
    assert got[0].kb_id == 5 and got[0].subtag_ids == [12, 13]
    assert got[1].kb_id == 7 and got[1].subtag_ids == []

@pytest.mark.parametrize("bad", [
    [{"kb_id": "five"}],
    [{"subtag_ids": [1]}],
    [{"kb_id": 1, "subtag_ids": [1, "two"]}],
    "not a list",
    [{"kb_id": 1, "subtag_ids": [1, 1]}],
])
def test_invalid_shapes_rejected(bad):
    with pytest.raises(ValueError):
        validate_selected_kb_config(bad)
