"""Test cases for the __main__ module."""
from unittest import mock
from unittest.mock import MagicMock

import click.testing
import eolt_root_cause_analyser.cli as cli
import pandas as pd
import pytest
from eolt_root_cause_analyser.sql_fetch import fetch_eol
from eolt_root_cause_analyser.tdms_fetch import form_filename_tdms
from eolt_root_cause_analyser.tdms_fetch import read_tdms
from pandas import DataFrame


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
    """Tests the fetch_eol function using monkeypatching to mock the pd.read_sql_query function.
    Looks to see that it can read a dataframe to the one useful value.

    Args:
        monkeypatch: The pytest monkeypatch fixture, used to temporarily replace the pd.read_sql_query function with a
            mock function.

    """
    # Set up mock return value
    mock_fetch_eol = mock.Mock(return_value=pd.DataFrame({"EOL_Test_ID": [123]}))

    # Monkey patch the pd.read_sql_query function
    monkeypatch.setattr(pd, "read_sql_query", mock_fetch_eol)

    # Call the function
    result = fetch_eol(1, 2)

    # Assert that the function returns the expected value
    assert result == 123


def test_fetch_eol_error(monkeypatch):
    """Tests the fetch_eol function using monkeypatching to mock the pd.read_sql_query function and raise an error.
    Checks that it can raise an error correctly.

    Args:
        monkeypatch: The pytest monkeypatch fixture, used to temporarily replace the pd.read_sql_query function with a
            mock function that raises an error.

    """
    # Set up mock side effect to raise an error
    mock_fetch_eol = mock.Mock(side_effect=Exception("Test error"))

    # Monkey patch the pd.read_sql_query function
    monkeypatch.setattr(pd, "read_sql_query", mock_fetch_eol)

    # Call the function and assert that an exception is raised
    with pytest.raises(Exception) as excinfo:
        result = fetch_eol(1, 2)
    assert str(excinfo.value) == "Test error"


def test_read_tdms(monkeypatch):
    """Tests the read_tdms function using monkeypatching to mock the yasa_file_io.tdms.read_tdms_as_dataframe function.
    Checks that yasa-file-io can indeed read a tdms to a pandas dataframe.

    Args:
        monkeypatch: The pytest monkeypatch fixture, used to temporarily replace the yasa_file_io.tdms.read_tdms_as_dataframe function with a mock function that returns a predefined DataFrame.

    """
    # Create a mock DataFrame to return from the patched function
    mock_df = DataFrame({"A": [1, 2], "B": [3, 4]})

    # Patch the yasa_file_io.tdms.read_tdms_as_dataframe function to return the mock DataFrame
    monkeypatch.setattr(
        "yasa_file_io.tdms.read_tdms_as_dataframe",
        MagicMock(return_value=mock_df),
    )

    # Call the function with a test filename
    result = read_tdms("test.tdms")

    # Assert that the result is equal to the mock DataFrame
    assert result.equals(mock_df)


def test_read_tdms_exception(monkeypatch):
    """Tests the read_tdms function using monkeypatching to mock the yasa_file_io.tdms.read_tdms_as_dataframe function
    and raise an exception.
    Checks that exceptions are as expected.

    Args:
        monkeypatch: The pytest monkeypatch fixture, used to temporarily replace the
            yasa_file_io.tdms.read_tdms_as_dataframe
            function with a mock function that raises an exception.

    """
    # Patch the yasa_file_io.tdms.read_tdms_as_dataframe function to raise an exception
    monkeypatch.setattr(
        "yasa_file_io.tdms.read_tdms_as_dataframe",
        MagicMock(side_effect=Exception("Test Exception")),
    )

    # Call the function with a test filename
    with pytest.warns(UserWarning, match="Accessing TDMS Failed"):
        result = read_tdms("test.tdms")

    # Assert that the result is None
    assert result is None


def test_form_filename_tdms():
    """Tests the form_filename_tdms function by calling it with test arguments and asserting that the result is the expected filename."""
    # Call the function with test arguments
    result = form_filename_tdms(1, 2, 3)

    # Assert that the result is the expected filename
    assert result == "3_2_1.tdms"
