import re
from typing import Any

from ai_toolkit._meta import __author__, __version__, display_banner, get_author, get_version


def test_version_format() -> None:
    """Test that version string follows semantic versioning."""

    version = __version__

    # Semantic versioning pattern: MAJOR.MINOR.PATCH
    pattern = r"^\d+\.\d+(\.\d+)?$"
    assert (
        re.match(pattern, version) is not None
    ), f"Version {version} doesn't follow semantic versioning"


def test_get_version() -> None:
    """Test get_version function."""

    # Test function returns correct version
    version = get_version()
    assert version == __version__
    assert isinstance(version, str)


def test_get_author() -> None:
    """Test get_author function."""

    # Test function returns correct author
    author = get_author()
    assert author == __author__
    assert isinstance(author, str)


def test_display_banner(capsys: Any) -> None:
    """Test display_banner function."""

    # Call the function
    display_banner()

    # Capture the output
    captured = capsys.readouterr()
    output = captured.out

    # Check that banner contains expected content
    assert "AI - TOOLKIT" in output
    assert f"Version: {__version__}" in output
    assert f"Author:  {__author__}" in output

    # Check that output is not empty
    assert len(output.strip()) > 0


def test_version_consistency() -> None:
    """Test version consistency across different access methods."""

    # Direct access
    version_attr = __version__

    # Function access
    version_func = get_version()

    # Compare
    assert version_attr == version_func

    # Type check
    assert isinstance(version_attr, str)
    assert isinstance(version_func, str)


def test_author_metadata() -> None:
    """Test author metadata."""

    assert isinstance(__author__, str)
    assert len(__author__) > 0


def test_author_consistency() -> None:
    """Test author consistency across different access methods."""

    # Direct access
    author_attr = __author__

    # Function access
    author_func = get_author()

    # Compare
    assert author_attr == author_func

    # Type check
    assert isinstance(author_attr, str)
    assert isinstance(author_func, str)


def test_banner_function_callable() -> None:
    """Test that display_banner function is callable."""

    assert callable(display_banner)


def test_module_constants() -> None:
    """Test that module constants have expected values."""

    # Test version format
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"

    # Test author format
    assert isinstance(__author__, str)
    assert __author__ == "Chrissi"


def test_banner_no_exception() -> None:
    """Test that display_banner doesn't raise exceptions."""

    # This should not raise any exceptions
    display_banner()
