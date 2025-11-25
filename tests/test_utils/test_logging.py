import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest

from ai_toolkit.base import LoggingConfig
from ai_toolkit.utils.logging import Logger, get_logger


@pytest.fixture
def temp_log_dir(tmp_path) -> Generator[Path, None, None]:
    """Create a temporary directory for log files.

    Args:
        tmp_path: Pytest fixture for temporary directory

    Yields:
        Path: Temporary log directory
    """

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    yield log_dir


class TestLoggerConfig:
    """Test suite for LoggerConfig."""

    def test_default_values(self):
        """Test default configuration values."""

        config = LoggingConfig()

        assert config.name == "logger"
        assert config.level == logging.INFO
        assert isinstance(config.dir, Path)
        assert config.enable_console is True
        assert config.enable_file is True

    def test_custom_values(self):
        """Test custom configuration values."""

        config = LoggingConfig(
            name="test_logger",
            level=logging.DEBUG,
            dir="custom_logs",
            enable_console=False,
            enable_file=False,
        )

        assert config.name == "test_logger"
        assert config.level == logging.DEBUG
        assert isinstance(config.dir, Path)
        assert str(config.dir) == "custom_logs"
        assert config.enable_console is False
        assert config.enable_file is False

    def test_dir_conversion(self):
        """Test DIR string to Path conversion."""

        # Test with string
        config = LoggingConfig(dir="test_logs")
        assert isinstance(config.dir, Path)
        assert str(config.dir) == "test_logs"

        # Test with Path
        path = Path("test_logs")
        config = LoggingConfig(dir=path)
        assert isinstance(config.dir, Path)
        assert config.dir == path


class TestLogger:
    """Test suite for Logger."""

    def test_initialization(self):
        """Test logger initialization."""

        config = LoggingConfig(name="test_logger")
        logger = Logger(config)

        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.INFO
        assert isinstance(logger.context, dict)
        assert len(logger.context) == 0

    def test_console_handler(self):
        """Test console handler setup."""

        config = LoggingConfig(name="test_logger", enable_console=True, enable_file=False)
        logger = Logger(config)

        handlers = logger.logger.handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)

    def test_file_handler(self, temp_log_dir):
        """Test file handler setup."""

        config = LoggingConfig(
            name="test_logger", dir=temp_log_dir, enable_console=False, enable_file=True
        )
        logger = Logger(config)

        handlers = logger.logger.handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.FileHandler)

        # Check if log file was created
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) == 1

    def test_both_handlers(self, temp_log_dir):
        """Test both console and file handlers."""

        config = LoggingConfig(
            name="test_logger", dir=temp_log_dir, enable_console=True, enable_file=True
        )
        logger = Logger(config)

        handlers = logger.logger.handlers
        assert len(handlers) == 2
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)
        assert any(isinstance(h, logging.FileHandler) for h in handlers)

    def test_context_management(self):
        """Test context management."""

        logger = Logger(LoggingConfig(name="test_logger"))

        # Set context
        logger.set_context(test_key="test_value")
        assert logger.context == {"test_key": "test_value"}

        # Add to context
        logger.set_context(another_key="another_value")
        assert logger.context == {
            "test_key": "test_value",
            "another_key": "another_value",
        }

        # Clear context
        logger.clear_context()
        assert len(logger.context) == 0

    def test_message_formatting(self):
        """Test message formatting."""

        logger = Logger(LoggingConfig(name="test_logger"))

        # Test without context or extra
        message = logger._format_message("test message", "INFO")
        data = json.loads(message)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert data["context"] == {}
        assert "extra" not in data

        # Test with context
        logger.set_context(test_key="test_value")
        message = logger._format_message("test message", "INFO")
        data = json.loads(message)
        assert data["context"] == {"test_key": "test_value"}

        # Test with extra
        message = logger._format_message("test message", "INFO", {"extra_key": "extra_value"})
        data = json.loads(message)
        assert data["extra"] == {"extra_key": "extra_value"}

    def test_log_levels(self, temp_log_dir):
        """Test different log levels."""

        config = LoggingConfig(
            name="test_logger",
            dir=temp_log_dir,
            enable_console=False,
            enable_file=True,
            level=logging.DEBUG,
        )
        logger = Logger(config)

        # Test all log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Check log file content
        log_file = next(temp_log_dir.glob("*.log"))
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert all(
                level in content for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            )

    def test_error_logging(self):
        """Test error logging with exception."""

        logger = Logger(LoggingConfig(name="test_logger"))

        try:
            raise ValueError("Test error")
        except Exception as e:
            message = logger._format_message("Error occurred", "ERROR", {"error": str(e)})
            data = json.loads(message)
            assert "error" in data["extra"]
            assert data["extra"]["error"] == "Test error"

    def test_serialize_dict_basic_types(self):
        """Test serialization of basic Python types."""

        logger = Logger(LoggingConfig())

        test_dict = {
            "int": 28,
            "float": 3.14,
            "str": "test",
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"a": 1},
        }

        result = logger._serialize_dict(test_dict)
        assert result == test_dict

    def test_serialize_dict_custom_objects(self):
        """Test serialization of custom objects."""

        logger = Logger(LoggingConfig())
        config = LoggingConfig(name="test", level=logging.DEBUG)
        now = datetime.now()

        class CustomString:
            __slots__ = ()

            def __str__(self):
                return "custom string representation"

        test_dict = {
            "config": config,
            "datetime": now,
            "str_obj": CustomString(),
            "mock": Mock(name="test_mock"),
            "range_obj": range(5),
            "complex_number": complex(1, 2),
        }

        result = logger._serialize_dict(test_dict)

        # Config should be serialized to dict
        assert isinstance(result["config"], dict)
        assert result["config"]["name"] == "test"
        assert result["config"]["level"] == logging.DEBUG

        # Datetime should be serialized to ISO format string
        assert isinstance(result["datetime"], str)
        assert result["datetime"] == now.isoformat()

        # Custom string object should be serialized to string
        assert isinstance(result["str_obj"], str)
        assert result["str_obj"] == "custom string representation"

        # Mock object should be serialized to dict
        assert isinstance(result["mock"], dict)
        assert len(result["mock"]) > 0

        # Range object should be serialized to string
        assert isinstance(result["range_obj"], str)
        assert result["range_obj"] == "range(0, 5)"

        # Complex number should be serialized to string
        assert isinstance(result["complex_number"], str)
        assert result["complex_number"] == "(1+2j)"

    def test_serialize_dict_non_serializable(self):
        """Test handling of non-serializable objects."""

        logger = Logger(LoggingConfig())

        # Object that raises an exception when converted to string
        class NonSerializable:
            def __str__(self):
                raise Exception("Can't convert to string")

            @property
            def __dict__(self):
                raise Exception("Can't get dict")

        test_dict = {"bad_object": NonSerializable()}

        result = logger._serialize_dict(test_dict)
        assert "<non-serializable: " in result["bad_object"]


def test_get_logger():
    """Test get_logger convenience function."""

    # Test with default config
    logger = get_logger("test_logger")
    assert isinstance(logger, Logger)
    assert logger.logger.name == "test_logger"

    # Test with custom config
    logger = get_logger(
        "test_logger",
        level=logging.DEBUG,
        enable_console=False,
        enable_file=False,
    )

    assert logger.logger.level == logging.DEBUG
    assert len(logger.logger.handlers) == 0


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after tests."""

    yield

    # Cleanup
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Remove all handlers from our test logger
    test_logger = logging.getLogger("test_logger")
    test_logger.handlers = []
