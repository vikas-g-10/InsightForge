import pytest

from settings import DEFAULT_ALLOWED_ORIGINS, ConfigError, Settings


def test_unsupported_mode_is_rejected_with_clear_message():
    with pytest.raises(ConfigError, match="APP_MODE"):
        Settings.from_env({"APP_MODE": "turbo"})


def test_blank_mode_falls_back_to_lite():
    assert Settings.from_env({"APP_MODE": "  "}).mode == "lite"


def test_default_origins_cover_vite_dev_server():
    assert "http://localhost:5173" in Settings.from_env({}).allowed_origins
    assert Settings.from_env({}).allowed_origins == DEFAULT_ALLOWED_ORIGINS


def test_origins_parsed_trimmed_and_normalized():
    s = Settings.from_env(
        {"ALLOWED_ORIGINS": " https://app.example.com/ , http://localhost:3000 "}
    )
    assert s.allowed_origins == ("https://app.example.com", "http://localhost:3000")


@pytest.mark.parametrize(
    "bad", ["*", "https://ok.example.com,*", "localhost:5173", "http://x.com/path", ","]
)
def test_invalid_origins_are_rejected(bad):
    with pytest.raises(ConfigError):
        Settings.from_env({"ALLOWED_ORIGINS": bad})


def test_settings_do_not_require_supabase_or_api_keys():
    assert Settings.from_env({}).mode == "lite"


def test_ingest_limit_defaults():
    s = Settings.from_env({})
    assert (s.max_rows, s.max_upload_bytes) == (2000, 50 * 1024 * 1024)


def test_ingest_limits_from_env():
    s = Settings.from_env({"MAX_ROWS": " 500 ", "MAX_UPLOAD_BYTES": "1048576"})
    assert (s.max_rows, s.max_upload_bytes) == (500, 1048576)


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "1.5"])
@pytest.mark.parametrize("name", ["MAX_ROWS", "MAX_UPLOAD_BYTES"])
def test_invalid_ingest_limits_are_rejected(name, bad):
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({name: bad})


def test_ollama_settings_defaults():
    s = Settings.from_env({})
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "gemma3:4b"
    assert s.ollama_timeout_seconds == 30.0
    assert s.max_image_upload_bytes == 10 * 1024 * 1024


def test_ollama_settings_from_env():
    s = Settings.from_env({
        "OLLAMA_BASE_URL": "http://192.168.1.50:11434/",
        "OLLAMA_MODEL": "llava:7b",
        "OLLAMA_TIMEOUT_SECONDS": "45.5",
        "MAX_IMAGE_UPLOAD_BYTES": "20971520",
    })
    assert s.ollama_base_url == "http://192.168.1.50:11434"
    assert s.ollama_model == "llava:7b"
    assert s.ollama_timeout_seconds == 45.5
    assert s.max_image_upload_bytes == 20971520


def test_invalid_ollama_base_url_rejected():
    with pytest.raises(ConfigError, match="OLLAMA_BASE_URL"):
        Settings.from_env({"OLLAMA_BASE_URL": "ftp://not-http"})
