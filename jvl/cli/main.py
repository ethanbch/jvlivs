import sys
import typer

# Force UTF-8 stdout/stderr on Windows to prevent 'charmap' UnicodeEncodeErrors with emojis
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from jvl.cli.ask import ask
from jvl.cli.chat import chat
from jvl.cli.config import config_app
from jvl.cli.default import default
from jvl.cli.history import history
from jvl.cli.model import model
from jvl.cli.review import review

app = typer.Typer(help="JVLIVS CLI")
app.command()(ask)
app.command()(chat)
app.command()(model)
app.command()(default)
app.command()(history)
app.command()(review)
app.add_typer(config_app, name="config")


@app.callback()
def main() -> None:
    """Point d'entrée principal de JVLIVS."""
    pass


