"""Pytest configuration — load .env so LLM API keys are available at collection time.

The smoke tests use `skipif` on the presence of an API key, which is evaluated during
collection. Loading the .env here (before test modules are collected) ensures the key is
visible to those markers.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass
