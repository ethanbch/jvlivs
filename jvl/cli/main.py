import typer

app = typer.Typer(help="JVLIVS CLI")


@app.callback()
def main() -> None:
    """Point d'entrée principal de JVLIVS."""
    pass
