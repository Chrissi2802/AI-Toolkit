"""
Meta information for AI Toolkit.
"""

import pyfiglet


__version__ = "0.1.0"
__author__ = "Chrissi"


def get_version() -> str:
    """Get the current version of AI-Toolkit.

    Returns:
        str: The current version of AI-Toolkit.
    """

    return __version__


def get_author() -> str:
    """Get the author of AI-Toolkit.

    Returns:
        str: The author of AI-Toolkit.
    """

    return __author__


def display_banner() -> None:
    """Display the banner for AI-Toolkit."""

    name = "AI - TOOLKIT"
    print(pyfiglet.figlet_format(name, font="standard"))
    print(name)
    print(f"Version: {get_version()}")
    print(f"Author:  {get_author()}")


if __name__ == "__main__":
    pass
