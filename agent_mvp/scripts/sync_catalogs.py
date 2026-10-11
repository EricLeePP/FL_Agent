"""Refresh the pinned STRONG AYA catalog inputs from their public repositories."""

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CATALOGS = ROOT / "src" / "federated_agent" / "data" / "catalogs"
SOURCES = {
    "AYA_cancer_schema.jsonld": (
        "https://raw.githubusercontent.com/STRONGAYA/"
        "AYA-cancer-semantic-map/main/AYA_cancer_schema.jsonld"
    ),
    "algorithm_store.json": (
        "https://raw.githubusercontent.com/STRONGAYA/"
        "v6-descriptive-statistics/v2.1.0/algorithm_store.json"
    ),
}


def main() -> None:
    target = CATALOGS
    target.mkdir(exist_ok=True)
    for filename, url in SOURCES.items():
        with urlopen(url, timeout=60) as response:
            content = response.read()
        (target / filename).write_bytes(content)
        print(f"Updated {filename} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
