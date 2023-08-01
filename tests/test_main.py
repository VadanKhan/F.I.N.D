"""Test cases for the __main__ module."""
from pathlib import Path
from unittest import mock

import click.testing
import eolt_root_cause_analyser.cli as cli
import pandas as pd
import pytest
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_eol
from eolt_root_cause_analyser.fetching.tdms_fetch import _apply_channel_map
from eolt_root_cause_analyser.fetching.tdms_fetch import read_tdms
from nptdms import TdmsFile


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
        fetch_eol(1, 2)
    assert str(excinfo.value) == "Test error"


def test_read_tdms(monkeypatch):
    # Create a mock TDMS file object
    class MockTdmsFile:
        @staticmethod
        def read(file_path):
            return MockTdmsFile()

        def as_dataframe(self):
            data = {"Ambient_new": [1, 2, 3]}
            return pd.DataFrame(data)

    # Monkeypatch the TdmsFile.read method to return the mock TDMS file object
    monkeypatch.setattr(TdmsFile, "read", MockTdmsFile.read)

    # Create a mock _apply_channel_map function that does nothing
    def mock_apply_channel_map(df, channel_map, fuzzy_matching, drop_duplicates, extract_all, fuzzy_match_score):
        return df

    # Monkeypatch the _apply_channel_map function to return the mock function
    monkeypatch.setattr("eolt_root_cause_analyser.fetching.tdms_fetch._apply_channel_map", mock_apply_channel_map)

    # Call the read_tdms function with a sample file path
    tdms_file_path = Path("test.tdms")
    df = read_tdms(tdms_file_path)

    # Check that the returned DataFrame has the expected data
    expected_data = {"Ambient_new": [1, 2, 3]}
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(df, expected_df)


def test_apply_channel_map():
    # Create a sample DataFrame
    data = {"/group1'/'channel1'": [1, 2, 3], "/group2'/'channel2'": [4, 5, 6]}
    df = pd.DataFrame(data)

    # Call the _apply_channel_map function with sample arguments
    channel_map = {"channel1": "new_channel1", "channel2": "new_channel2"}
    result = _apply_channel_map(df, channel_map, False, False)

    # Check that the returned DataFrame has the expected data
    expected_data = {"new_channel1": [1, 2, 3], "new_channel2": [4, 5, 6]}
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected_df)
