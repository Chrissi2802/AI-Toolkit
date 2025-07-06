import re

import ai_toolkit


def test_version_format():
    """Test that version string follows semantic versioning."""

    version = ai_toolkit.__version__

    # Semantic versioning pattern: MAJOR.MINOR.PATCH
    pattern = r"^\d+\.\d+(\.\d+)?$"
    assert (
        re.match(pattern, version) is not None
    ), f"Version {version} doesn't follow semantic versioning"


def test_get_version():
    """Test get_version function."""

    # Test that function exists
    assert hasattr(ai_toolkit, "get_version")
    assert callable(ai_toolkit.get_version)

    # Test function returns correct version
    version = ai_toolkit.get_version()
    assert version == ai_toolkit.__version__
    assert isinstance(version, str)


def test_get_author():
    """Test get_author function."""

    # Test that function exists
    assert hasattr(ai_toolkit, "get_author")
    assert callable(ai_toolkit.get_author)

    # Test function returns correct author
    author = ai_toolkit.get_author()
    assert author == ai_toolkit.__author__
    assert isinstance(author, str)


def test_display_banner():
    """Test display_banner function."""

    # Test that function exists
    assert hasattr(ai_toolkit, "display_banner")
    assert callable(ai_toolkit.display_banner)


def test_version_consistency():
    """Test version consistency across different access methods."""

    # Direct access
    version_attr = ai_toolkit.__version__

    # Function access
    version_func = ai_toolkit.get_version()

    # Compare
    assert version_attr == version_func

    # Type check
    assert isinstance(version_attr, str)
    assert isinstance(version_func, str)


def test_author_metadata():
    """Test author metadata."""

    assert hasattr(ai_toolkit, "__author__")
    assert isinstance(ai_toolkit.__author__, str)
    assert len(ai_toolkit.__author__) > 0


def test_import_structure():
    """Test the import structure of the package."""

    # Test that submodules are properly structured
    import ai_toolkit.base
    import ai_toolkit.models
    import ai_toolkit.training
    import ai_toolkit.utils

    # Test that each submodule has an __init__.py
    assert hasattr(ai_toolkit.base, "__file__")
    assert hasattr(ai_toolkit.models, "__file__")
    assert hasattr(ai_toolkit.training, "__file__")
    assert hasattr(ai_toolkit.utils, "__file__")


def test_documentation_strings():
    """Test that main components have documentation."""

    assert ai_toolkit.__doc__ is not None
    assert ai_toolkit.base.__doc__ is not None
    assert ai_toolkit.models.__doc__ is not None
    assert ai_toolkit.training.__doc__ is not None
    assert ai_toolkit.utils.__doc__ is not None
