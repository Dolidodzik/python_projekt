import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exploration.plots import DiagramGenerator


class DiagramRunner:

    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        dataset_name: str = "titanic",
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name

    def validate(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Brak pliku danych: {self.csv_path}")

    def run(self) -> dict:
        self.validate()
        generator = DiagramGenerator.from_csv(
            self.csv_path,
            self.output_dir,
            dataset_name=self.dataset_name,
        )
        return generator.generate_all()

    def print_report(self, result: dict) -> None:
        print(f"Wczytywanie: {self.csv_path}")
        print(f"Zapis diagramów do: {self.output_dir}")
        print("\nWygenerowane pliki:")
        for name, path in result["files"].items():
            print(f"  - {name}: {path}")
        print("\nTypy kolumn wykryte automatycznie:")
        print(json.dumps(result["column_types"], indent=2, ensure_ascii=False))


def main():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent

    runner = DiagramRunner(
        csv_path=project_root / "Titanic-Dataset.csv",
        output_dir=base_dir / "output" / "diagramy",
        dataset_name="titanic",
    )

    try:
        result = runner.run()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    runner.print_report(result)


if __name__ == "__main__":
    main()
