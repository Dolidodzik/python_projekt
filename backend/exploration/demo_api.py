import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import httpx
except ImportError:
    print("Zainstaluj httpx: pip install httpx")
    sys.exit(1)


class ApiDemoClient:

    def __init__(
        self,
        csv_path: Path,
        base_url: str = "http://localhost:8000/exploration",
        server_url: str = "http://localhost:8000/",
        timeout: float = 30.0,
    ):
        self.csv_path = csv_path
        self.base_url = base_url.rstrip("/")
        self.server_url = server_url
        self.timeout = timeout
        self._scenarios: List[Tuple[str, str, Dict]] = [
            ("Typy kolumn", "/column-types", {}),
            ("Statystyki (Age, Fare)", "/basic-stats", {"columns": "Age,Fare"}),
            ("Korelacja", "/correlation", {"columns": "Age,Fare,SibSp,Parch", "method": "pearson"}),
            ("Rozkład Age", "/distribution", {"column": "Age"}),
            ("PCA", "/pca", {"columns": "Age,Fare,SibSp,Parch", "n_components": 3}),
            ("Raport zbiorczy", "/describe-all", {}),
        ]

    def validate_csv(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Brak pliku: {self.csv_path}")

    def check_server(self) -> None:
        try:
            httpx.get(self.server_url, timeout=3.0)
        except httpx.ConnectError as e:
            raise ConnectionError("Serwer nie działa. Uruchom: uvicorn main:app --reload") from e

    def call_endpoint(self, name: str, path: str, data: dict) -> None:
        url = f"{self.base_url}{path}"
        with open(self.csv_path, "rb") as f:
            files = {"file": (self.csv_path.name, f, "text/csv")}
            response = httpx.post(url, data=data, files=files, timeout=self.timeout)

        print(f"\n=== {name} ===")
        print(f"POST {url}")
        if response.status_code != 200:
            print(f"Błąd {response.status_code}: {response.text}")
            return

        body = response.json()
        serialized = json.dumps(body, indent=2, ensure_ascii=False)
        print(serialized[:1500])
        if len(serialized) > 1500:
            print("... (wynik skrócony)")

    def run_all(self) -> None:
        self.validate_csv()
        self.check_server()
        for name, path, data in self._scenarios:
            self.call_endpoint(name, path, data)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    client = ApiDemoClient(csv_path=project_root / "Titanic-Dataset.csv")

    try:
        client.run_all()
    except (FileNotFoundError, ConnectionError) as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
