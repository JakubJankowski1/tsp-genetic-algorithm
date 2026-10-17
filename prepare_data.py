"""Download the three TSPLIB benchmarks used by this project.

Run this script locally. The downloaded data directory is excluded from Git.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SOURCE = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp"
INSTANCES = {"berlin52": 52, "kroA100": 100, "kroA150": 150}


def count_coordinates(content: str) -> int:
    in_section = False
    count = 0
    for line in content.splitlines():
        item = line.strip()
        if item == "NODE_COORD_SECTION":
            in_section = True
        elif item == "EOF":
            break
        elif in_section and item:
            count += 1
    return count


def first_11_from_berlin52(content: str) -> str:
    """Reproduce the original project's custom 11-city subset locally."""
    lines = content.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "NODE_COORD_SECTION")
    coordinates = lines[start + 1 : start + 12]
    if len(coordinates) != 11 or any(len(line.split()) < 3 for line in coordinates):
        raise ValueError("berlin52 does not contain the expected first 11 coordinates")
    return "\n".join(
        [
            "NAME: berlin11_modified",
            "TYPE: TSP",
            "COMMENT: First 11 locations from berlin52; locally derived subset",
            "DIMENSION: 11",
            "EDGE_WEIGHT_TYPE: EUC_2D",
            "NODE_COORD_SECTION",
            *coordinates,
            "EOF",
            "",
        ]
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, expected_count in INSTANCES.items():
        destination = DATA_DIR / f"{name}.tsp"
        if destination.exists():
            content = destination.read_text(encoding="utf-8")
        else:
            url = f"{SOURCE}/{name}.tsp.gz"
            print(f"Downloading {url}")
            with urlopen(url, timeout=30) as response:
                content = gzip.decompress(response.read()).decode("utf-8")
        if count_coordinates(content) != expected_count:
            raise ValueError(f"Unexpected coordinate count in {name}.tsp")
        destination.write_text(content, encoding="utf-8")
        print(f"Ready: {destination.name} ({expected_count} cities)")

        if name == "berlin52":
            subset = DATA_DIR / "berlin11_modified.tsp"
            subset.write_text(first_11_from_berlin52(content), encoding="utf-8")
            print(f"Ready: {subset.name} (11 cities, derived from berlin52)")


if __name__ == "__main__":
    main()
