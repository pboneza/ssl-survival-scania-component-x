from pathlib import Path
import nbformat


NOTEBOOK_DIR = Path("notebooks")

MINIMAL_METADATA = {
    "kernelspec": {
        "display_name": "Python (ssl-survival-scania)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def clean_notebook(notebook_path: Path) -> None:
    """Remove outputs and unnecessary metadata from a notebook."""
    notebook = nbformat.read(notebook_path, as_version=4)

    notebook.metadata = MINIMAL_METADATA.copy()

    for cell in notebook.cells:
        cell.metadata = {}

        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)

    print(f"Cleaned: {notebook_path}")


def main() -> None:
    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

    if not notebook_paths:
        raise FileNotFoundError("No notebooks found in the notebooks/ directory.")

    for notebook_path in notebook_paths:
        clean_notebook(notebook_path)

    print("All notebooks cleaned and validated successfully.")


if __name__ == "__main__":
    main()