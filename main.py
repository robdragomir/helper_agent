#!/usr/bin/env python
"""
Main entry point for the LangGraph Helper Agent.
"""

import sys
import os


# Ensure the current directory is in the path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to file only (before importing other modules)
from app.core.logging_config import configure_logging
configure_logging()

from app.presentation.cli import main

if __name__ == "__main__":
    main()