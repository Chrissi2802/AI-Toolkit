import ai_toolkit


def test_import_structure() -> None:
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


def test_documentation_strings() -> None:
    """Test that main components have documentation."""

    assert ai_toolkit.__doc__ is not None
    assert ai_toolkit.base.__doc__ is not None
    assert ai_toolkit.models.__doc__ is not None
    assert ai_toolkit.training.__doc__ is not None
    assert ai_toolkit.utils.__doc__ is not None
