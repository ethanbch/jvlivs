import typer

from jvl.cli.ask import ask
from jvl.cli.chat import chat
from jvl.cli.config import config_app
from jvl.cli.default import default
from jvl.cli.model import model

app = typer.Typer(help="JVLIVS CLI")
app.command()(ask)
app.command()(chat)
app.command()(model)
app.command()(default)
app.add_typer(config_app, name="config")


@app.callback()
def main() -> None:
    """Point d'entrée principal de JVLIVS."""
    pass

