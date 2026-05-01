"""Console script for pykawa."""

import typer
from rich.console import Console

from pykawa import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for pykawa."""
    console.print("Replace this message by putting your code into pykawa.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
