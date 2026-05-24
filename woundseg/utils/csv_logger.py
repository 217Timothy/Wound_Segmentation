"""Minimal append-only CSV logger for per-epoch metrics."""

import csv
from pathlib import Path


class CSVLogger:
    """Writes one row per epoch to a CSV file, creating the header once."""

    def __init__(self, path: str | Path, columns: list[str]):
        self.path = Path(path)
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(columns)

    def append(self, row: dict) -> None:
        """Append a row; `row` keys must cover `self.columns`."""
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in self.columns])
