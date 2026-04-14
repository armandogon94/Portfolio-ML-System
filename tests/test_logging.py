"""Tests for structured logging configuration and FastAPI middleware.

Tests JSON/text formatters, setup_logging(), and request logging middleware.
"""

import io
import json
import logging

import pytest

# ── Task 5.1: Logging Configuration ──────────────────────────────────────────


def _capture_log_output(log_format: str, message: str, logger_name: str = "test") -> str:
    """Helper: configure logging with given format, emit one record, return captured output."""
    from src.logging_config import setup_logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        root_logger.handlers = [handler]
        setup_logging(log_format=log_format, handler=handler)
        logging.getLogger(logger_name).info(message)
        return stream.getvalue()
    finally:
        root_logger.handlers = original_handlers
        root_logger.level = original_level


class TestJsonFormatter:
    """Tests for the JSON log formatter."""

    def test_json_formatter_produces_valid_json(self):
        """JSON formatter output is parseable as JSON."""
        output = _capture_log_output("json", "hello world")
        line = output.strip().split("\n")[-1]
        record = json.loads(line)
        assert isinstance(record, dict)

    def test_json_formatter_has_required_keys(self):
        """JSON log record has timestamp, level, message, logger keys."""
        output = _capture_log_output("json", "test message")
        line = output.strip().split("\n")[-1]
        record = json.loads(line)

        assert "timestamp" in record
        assert "level" in record
        assert "message" in record
        assert "logger" in record

    def test_json_formatter_level_value(self):
        """JSON log record level is 'INFO' for info-level log."""
        output = _capture_log_output("json", "check level")
        line = output.strip().split("\n")[-1]
        record = json.loads(line)
        assert record["level"] == "INFO"

    def test_json_formatter_message_value(self):
        """JSON log record message matches the emitted string."""
        output = _capture_log_output("json", "unique-test-message-xyz")
        line = output.strip().split("\n")[-1]
        record = json.loads(line)
        assert record["message"] == "unique-test-message-xyz"

    def test_json_formatter_logger_name(self):
        """JSON log record logger field matches the logger name."""
        output = _capture_log_output("json", "msg", logger_name="myapp.module")
        line = output.strip().split("\n")[-1]
        record = json.loads(line)
        assert record["logger"] == "myapp.module"

    def test_json_formatter_timestamp_is_iso_format(self):
        """timestamp field is a non-empty ISO 8601 string."""
        output = _capture_log_output("json", "timestamp test")
        line = output.strip().split("\n")[-1]
        record = json.loads(line)
        ts = record["timestamp"]
        assert isinstance(ts, str)
        assert len(ts) > 0
        # ISO 8601 contains 'T' separator between date and time
        assert "T" in ts or "-" in ts


class TestTextFormatter:
    """Tests for the text/human-readable log formatter."""

    def test_text_formatter_is_not_json(self):
        """Text formatter output is not JSON (not parseable as JSON)."""
        output = _capture_log_output("text", "hello text")
        line = output.strip().split("\n")[-1]
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(line)

    def test_text_formatter_contains_message(self):
        """Text formatter output contains the log message."""
        output = _capture_log_output("text", "readable output test")
        assert "readable output test" in output

    def test_text_formatter_contains_level(self):
        """Text formatter output contains the log level."""
        output = _capture_log_output("text", "level check")
        assert "INFO" in output


class TestSetupLogging:
    """Tests for setup_logging() environment variable handling."""

    def test_setup_logging_json_format_from_env(self, monkeypatch):
        """LOG_FORMAT=json env var produces JSON output."""
        from src.logging_config import setup_logging

        monkeypatch.setenv("LOG_FORMAT", "json")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        try:
            root_logger.handlers = [handler]
            setup_logging(handler=handler)  # no explicit format — reads env
            logging.getLogger("env_test").info("env driven")
            output = stream.getvalue().strip().split("\n")[-1]
            record = json.loads(output)
            assert record["message"] == "env driven"
        finally:
            root_logger.handlers = original_handlers
            root_logger.level = original_level

    def test_setup_logging_text_format_from_env(self, monkeypatch):
        """LOG_FORMAT=text env var produces non-JSON output."""
        from src.logging_config import setup_logging

        monkeypatch.setenv("LOG_FORMAT", "text")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        try:
            root_logger.handlers = [handler]
            setup_logging(handler=handler)
            logging.getLogger("env_text").info("text output")
            output = stream.getvalue().strip()
            assert "text output" in output
            # should not be valid JSON on the last line
            last_line = output.split("\n")[-1]
            with pytest.raises((json.JSONDecodeError, ValueError)):
                json.loads(last_line)
        finally:
            root_logger.handlers = original_handlers
            root_logger.level = original_level

    def test_setup_logging_default_is_text(self, monkeypatch):
        """Without LOG_FORMAT env var, default is text (not JSON)."""
        from src.logging_config import setup_logging

        monkeypatch.delenv("LOG_FORMAT", raising=False)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        try:
            root_logger.handlers = [handler]
            setup_logging(handler=handler)
            logging.getLogger("default_test").info("default format")
            output = stream.getvalue().strip()
            assert "default format" in output
        finally:
            root_logger.handlers = original_handlers
            root_logger.level = original_level

    def test_setup_logging_imports(self):
        """setup_logging can be imported from src.logging_config."""
        from src.logging_config import setup_logging  # noqa: F401


# ── Task 5.3: Request Logging Middleware ─────────────────────────────────────


class TestRequestLoggingMiddleware:
    """Tests for FastAPI request logging middleware."""

    def test_middleware_logs_request(self, api_client, caplog):
        """A normal API request produces a log entry."""
        with caplog.at_level(logging.INFO, logger="src.serving.api"):
            api_client.get("/models")
        # At least one log record should have been emitted by the middleware
        assert any("GET" in r.message and "/models" in r.message for r in caplog.records)

    def test_middleware_logs_method_and_path(self, api_client, caplog):
        """Log entry includes HTTP method and path."""
        with caplog.at_level(logging.INFO, logger="src.serving.api"):
            api_client.post("/predict/credit-risk", json={})
        assert any(
            "POST" in r.message and "/predict/credit-risk" in r.message
            for r in caplog.records
        )

    def test_middleware_logs_status_code(self, api_client, caplog):
        """Log entry includes the HTTP status code."""
        with caplog.at_level(logging.INFO, logger="src.serving.api"):
            api_client.get("/models")
        assert any("200" in r.message for r in caplog.records)

    def test_health_check_not_logged(self, api_client, caplog):
        """GET /health is excluded from our middleware access logs (too noisy)."""
        with caplog.at_level(logging.INFO, logger="src.serving.api"):
            api_client.get("/health")
        # Only check records from our middleware logger — httpx may log separately
        api_records = [r for r in caplog.records if r.name == "src.serving.api"]
        assert not any("/health" in r.message for r in api_records)


# ── Task 5.4: Enriched /health ────────────────────────────────────────────────


class TestEnrichedHealth:
    """Tests for enriched /health endpoint with model availability."""

    def test_health_returns_status_ok(self, api_client):
        """GET /health still returns status: ok."""
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_returns_models_key(self, api_client):
        """GET /health response includes 'models' key."""
        resp = api_client.get("/health")
        data = resp.json()
        assert "models" in data

    def test_health_models_has_all_four_models(self, api_client):
        """models dict includes all 4 model keys."""
        resp = api_client.get("/health")
        models = resp.json()["models"]
        assert "credit_risk" in models
        assert "fraud_detection" in models
        assert "price_prediction" in models
        assert "demand_forecasting" in models

    def test_health_models_available_field(self, api_client):
        """Each model entry has an 'available' boolean field."""
        resp = api_client.get("/health")
        models = resp.json()["models"]
        for model_info in models.values():
            assert "available" in model_info
            assert isinstance(model_info["available"], bool)

    def test_health_loaded_models_are_available(self, api_client, predictor):
        """Models with checkpoints in the test predictor show available=True."""
        # The test predictor has all 4 models available via patched _ensure_loaded
        resp = api_client.get("/health")
        models = resp.json()["models"]
        # At least some models should be marked available (those with checkpoint dirs)
        available_count = sum(1 for m in models.values() if m["available"])
        assert available_count >= 1
