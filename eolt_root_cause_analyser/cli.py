import click
from eolt_root_cause_analyser.logic_toplevel import logic


@click.command()
@click.option(
    "--failure_code",
    nargs=1,
    type=str,
    default="failure_code",
    help="Give failure code in following format (after root_cause): '--failure_code [your input]'",
)
@click.option(
    "--test_id",
    nargs=1,
    type=str,
    default="test_id",
    help="Give test ID in following format (after root_cause): '--test_id [your input]'",
)
@click.option(
    "--test_type_id",
    nargs=1,
    type=str,
    default="test_type_id",
    help="Give test type ID in following format (after root_cause): '--test_type_id [your input]'",
)
def begin(failure_code, test_id, test_type_id):
    click.echo(f"received inputs: failure code={failure_code}, test id={test_id}, test type id={test_type_id}")
    logic(failure_code, test_id, test_type_id)
    return failure_code, test_id, test_type_id
