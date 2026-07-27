import pytest

from audio_trombone.tools.validate_mdx23c import normalise_state_dict


def test_accepts_raw_state_dict():
    tensor = object()
    assert normalise_state_dict({"weight": tensor}) == {"weight": tensor}


@pytest.mark.parametrize("wrapper", ["state_dict", "model"])
def test_accepts_documented_wrapper_and_uniform_module_prefix(wrapper):
    tensor = object()
    result = normalise_state_dict({wrapper: {"module.weight": tensor}})
    assert result == {"weight": tensor}


def test_rejects_ambiguous_wrappers():
    with pytest.raises(ValueError, match="ambiguous"):
        normalise_state_dict({"state_dict": {"a": 1}, "model": {"a": 1}})


def test_rejects_mixed_module_prefixes():
    with pytest.raises(ValueError, match="mixed"):
        normalise_state_dict({"module.weight": 1, "bias": 2})


def test_rejects_empty_state_dict():
    with pytest.raises(ValueError, match="non-empty"):
        normalise_state_dict({})
