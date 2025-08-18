import json
import logging
import sys
from datetime import date, datetime
from typing import Any, Dict, Optional

from ai_toolkit.base.config import ConfigFactory, LoggingConfig


class Logger:
    """Logger class for logging messages to a file."""

    def __init__(self, config: LoggingConfig) -> None:
        """Initialize the logger.

        Args:
            config (LoggingConfig): Logger configurations.
        """

        self.config = config
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(self.config.level)
        self.formatter = logging.Formatter(self.config.format)
        self.context: Dict[str, Any] = {}

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler
        if self.config.enable_console:
            self._setup_console_handler()

        # File handler
        if self.config.enable_file:
            self._setup_file_handler()

    def _setup_console_handler(self) -> None:
        """Setup the console handler."""

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def _setup_file_handler(self) -> None:
        """Setup the file handler."""

        self.config.dir.mkdir(parents=True, exist_ok=True)

        # Date and time format
        # log_file = (
        #     self.config.dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}{self.config.file}"
        # )

        # Date format
        log_file = self.config.dir / f"{datetime.now():%Y-%m-%d}{self.config.file}"

        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def set_context(self, **kwargs: Any) -> None:
        """Set context for logging.

        Args:
            **kwargs: Context key-value pairs
        """

        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear the current context."""

        self.context.clear()

    def _format_message(
        self, message: str, level: str, extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format message with context and extra information.

        Args:
            message (str): Log message
            level (str): Log level
            extra (Optional[Dict[str, Any]], optional): Extra information.
                Defaults to None.

        Returns:
            str: Formatted message as JSON
        """

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "context": self.context,
        }

        if extra:
            log_data["extra"] = self._serialize_dict(extra)

        return json.dumps(log_data)

    def _serialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dict values to JSON serializable format.

        Args:
            d (Dict[str, Any]): Dictionary to serialize.

        Returns:
            Dict[str, Any]: Serialized dictionary.
        """

        result = {}

        for key, value in d.items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                result[key] = value
            except TypeError:
                try:
                    if hasattr(value, "__dict__"):
                        result[key] = vars(value)
                    elif isinstance(value, (datetime, date)):
                        result[key] = value.isoformat()
                    elif hasattr(value, "__str__"):
                        result[key] = str(value)
                    else:
                        result[key] = repr(value)
                except Exception:
                    result[key] = f"<non-serializable: {type(value).__name__}>"

        return result

    def _log(
        self, level: int, message: str, error: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Internal logging method.

        Args:
            level (int): Logging level
            message (str): Log message
            error (Optional[Exception], optional): Exception object. Defaults to None.
            **kwargs: Additional logging information
        """

        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)

        level_name = logging.getLevelName(level)
        formatted = self._format_message(message, level_name, kwargs)

        self.logger.log(level, formatted)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message (str): Debug message
            **kwargs: Additional logging information
        """

        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message (str): Info message
            **kwargs: Additional logging information
        """

        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message (str): Warning message
            **kwargs: Additional logging information
        """

        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message (str): Error message
            error (Optional[Exception], optional): Exception object. Defaults to None.
            **kwargs: Additional logging information
        """

        self._log(logging.ERROR, message, error, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log critical message.

        Args:
            message (str): Critical message
            error (Optional[Exception], optional): Exception object. Defaults to None.
            **kwargs: Additional logging information
        """

        self._log(logging.CRITICAL, message, error, **kwargs)


def get_logger(name: str, **config_kwargs: Any) -> Logger:
    """Convenience function to create a logger with custom configuration.

    Args:
        name (str): Logger name.
        **config_kwargs: Override default configuration.

    Returns:
        Logger: Configured logger instance.
    """

    config_factory = ConfigFactory()
    configs = config_factory.get_config()
    config = configs.logging
    config.name = name

    config_updates = {"name": name, **config_kwargs}
    config = config.model_copy(update=config_updates)

    return Logger(config)


if __name__ == "__main__":
    pass
