"""Test cases for the __main__ module."""
from unittest import mock

import click.testing
import eolt_root_cause_analyser.cli as cli


def test_begin_prints_error_too_many():
    """This tests that an error message is printed when there are too many args."""
    runner = click.testing.CliRunner()
    with mock.patch("builtins.print") as print:
        runner.invoke(cli.begin, ["--info", "string1, string2, string3, string4"])
        print.assert_called_once_with("Error: Too many arguments")
