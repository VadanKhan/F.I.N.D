"""Test cases for the __main__ module."""
import pytest

from eolt_root_cause_analyser import __main__


def test_main_succeeds() -> None:
    """Main exits with no exceptions."""
    error = None
    try:
        __main__.main()
    except Exception as e:
        error = e
    assert not error
        
