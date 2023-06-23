"""Test cases for the __main__ module."""
from unittest import mock

import click.testing
import eolt_root_cause_analyser.cli as cli


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
        print.assert_called_once_with("received inputs: failure_code=string1, test_id=string2, test_type_id=string3")
