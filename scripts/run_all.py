"""Run the full pipeline: generate data -> train all models -> evaluate."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

console = Console()

SCRIPTS_DIR = Path(__file__).parent


def run_step(description: str, cmd: list[str]) -> None:
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]{description}[/bold blue]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent)
    if result.returncode != 0:
        console.print(f"[bold red]Failed: {description}[/bold red]")
        sys.exit(1)


def main():
    console.print("[bold]Full ML Pipeline[/bold]\n")

    run_step("Step 1: Generate Synthetic Data", [sys.executable, str(SCRIPTS_DIR / "generate_data.py"), "--problem", "all"])
    run_step("Step 2: Train All Models", [sys.executable, str(SCRIPTS_DIR / "train.py"), "--model", "all", "--no-wandb"])
    run_step("Step 3: Evaluate and Summarize", [sys.executable, str(SCRIPTS_DIR / "evaluate.py")])

    console.print("\n[bold green]Full pipeline complete![/bold green]")
    console.print("Run [cyan]make ui[/cyan] to launch the web interface.")


if __name__ == "__main__":
    main()
