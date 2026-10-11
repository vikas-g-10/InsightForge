import pytest

from pipeline._common import PipelineConfigError
from pipeline.language import (
    EN, HI, HINGLISH, KANGLISH, TANGLISH, TENGLISH, MIXED, UNKNOWN,
    LANGUAGE_CODES, LANGUAGE_BADGES, language_code, load_language_config, tag_language, evaluate_pack,
)


@pytest.mark.parametrize(
    "text",
    ["The app is great and works very well", "Great app", "Battery drains quickly after the update"],
)
def test_english(text):
    assert language_code(text) == EN


@pytest.mark.parametrize("text", ["Yeh app bahut accha hai", "bahut kharab hai", "Delivery nahi hui aur paisa wapas nahi mila"])
def test_romanized_hinglish(text):
    assert language_code(text) == HINGLISH


@pytest.mark.parametrize("text", ["ऐप बहुत अच्छा है", "यह बेकार है और महंगा भी"])
def test_devanagari_hindi(text):
    assert language_code(text) == HI


def test_mixed_language_text():
    assert language_code("App is good but delivery bahut slow hai") == HINGLISH
    assert language_code("ऐप is really good hai") == HINGLISH  # Devanagari + Latin


@pytest.mark.parametrize(
    "text",
    [
        "camera tumba chennagide",  # Kannada (Romanized)
        "display chanda ide",
        "sound sakath agide",
    ],
)
def test_romanized_kanglish(text):
    assert language_code(text) == KANGLISH


@pytest.mark.parametrize(
    "text",
    [
        "phone romba waste",  # Tamil (Romanized)
        "camera sema nalla irukku",
        "battery romba mokka",
    ],
)
def test_romanized_tanglish(text):
    assert language_code(text) == TANGLISH


@pytest.mark.parametrize(
    "text",
    [
        "camera chala bagundi",  # Telugu (Romanized)
        "display keka undhi",
        "battery backup thopu",
    ],
)
def test_romanized_tenglish(text):
    assert language_code(text) == TENGLISH


def test_language_badges():
    assert tag_language("camera tumba chennagide").badge == "Kanglish"
    assert tag_language("phone romba waste").badge == "Tanglish"
    assert tag_language("camera chala bagundi").badge == "Tenglish"
    assert tag_language("Yeh app bahut accha hai").badge == "Hinglish"
    assert tag_language("The app is great").badge == "English"
    assert tag_language("ऐप बहुत अच्छा है").badge == "Hindi"


@pytest.mark.parametrize("text", [None, "", "   ", "😀😀", "12345 !!!"])
def test_empty_or_letterless_is_unknown(text):
    result = tag_language(text)
    assert result.code == UNKNOWN and result.tokens == 0


@pytest.mark.parametrize(
    "text",
    [
        "El producto es muy malo",  # Spanish
        "Das ist sehr schlecht",  # German
        "ಈ ಅಪ್ಲಿಕೇಶನ್ ಚೆನ್ನಾಗಿದೆ",  # Kannada: not supported
        "இந்த செயலி நன்றாக உள்ளது",  # Tamil: not supported
        "xyz qwerty",  # ambiguous gibberish
        "这个应用很好",  # Chinese
    ],
)
def test_unsupported_or_ambiguous_is_unknown(text):
    assert language_code(text) == UNKNOWN


def test_codes_are_documented_values():
    assert set(LANGUAGE_CODES) == {EN, HI, HINGLISH, KANGLISH, TANGLISH, TENGLISH, MIXED, UNKNOWN}
    assert language_code("Some random sentence about the app is here") in LANGUAGE_CODES


def test_deterministic():
    text = "App accha hai but slow hai"
    assert len({tag_language(text) for _ in range(5)}) == 1


def test_overlapping_marker_lists_are_rejected(tmp_path):
    bad = tmp_path / "markers.yaml"
    bad.write_text("hinglish_markers: [hai, app]\nenglish_words: [app, the]\n", encoding="utf-8")
    with pytest.raises(PipelineConfigError, match="app"):
        load_language_config(bad)


def test_custom_config_is_used(tmp_path):
    custom = tmp_path / "markers.yaml"
    custom.write_text("hinglish_markers: [zorp]\nenglish_words: [blah]\n", encoding="utf-8")
    cfg = load_language_config(custom)
    assert language_code("zorp zorp", cfg) == HINGLISH
    assert language_code("blah blah", cfg) == EN


@pytest.mark.parametrize("pack_code", ["kanglish", "tanglish", "tenglish"])
def test_pack_evaluation_benchmarks(pack_code):
    metrics = evaluate_pack(pack_code)
    assert metrics["total"] == 30
    assert metrics["detection_acc"] >= 0.70
    assert metrics["sentiment_acc"] >= 0.70
