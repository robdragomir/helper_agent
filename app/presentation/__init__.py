"""
Presentation layer - handles user interface and output formatting.
Follows onion architecture - outermost layer for user interaction.
"""

from .cli import app, main

__all__ = [
    "app",
    "main",
]