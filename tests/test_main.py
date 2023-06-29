"""Test cases for the __main__ module."""
from unittest import mock

import click.testing
import eolt_root_cause_analyser.cli as cli
import pandas as pd
import pytest
from eolt_root_cause_analyser.sql_fetch import fetch_eol


def test_begin_correct_printing():
    """This tests the correcting echoing from the cli base script"""
    runner = click.testing.CliRunner()
    with mock.patch("click.echo") as print:
        runner.invoke(
            cli.begin,
            [
                "--failure_code",
                "string1",
                "--test_id",
                "string2",
                "--test_type_id",
                "string3",
            ],
        )
        print.assert_called_once_with("received inputs: failure code=string1, test id=string2, test type id=string3")


def test_fetch_eol_success(monkeypatch):
    # Set up mock return value
    mock_fetch_eol = mock.Mock(return_value=pd.DataFrame({"EOL_Test_ID": [123]}))

    # Monkey patch the pd.read_sql_query function
    monkeypatch.setattr(pd, "read_sql_query", mock_fetch_eol)

    # Call the function
    result = fetch_eol(1, 2)

    # Assert that the function returns the expected value
    assert result == 123


def test_fetch_eol_error(monkeypatch):
    # Set up mock side effect to raise an error
    mock_fetch_eol = mock.Mock(side_effect=Exception("Test error"))

    # Monkey patch the pd.read_sql_query function
    monkeypatch.setattr(pd, "read_sql_query", mock_fetch_eol)

    # Call the function and assert that an exception is raised
    with pytest.raises(Exception) as excinfo:
        result = fetch_eol(1, 2)
    assert str(excinfo.value) == "Test error"
