import click


@click.command()
@click.option(
    "--info",
    default="failure_code,test_id,test_type",
    help="Give 3 bits of info in following format (after hook_trigger): --info=failure_code,test_id,test_type",
)
def begin(info):
    """Simple program that prints out the input information as you call this cmd interface"""
    click.echo(f"I am doing something magic with bits of info: {info}!")
    info_list = info.split(",")
    if len(info_list) > 3:
        print("Error: Too many arguments")
    if len(info_list) < 3:
        print("Error: Too few arguments")
