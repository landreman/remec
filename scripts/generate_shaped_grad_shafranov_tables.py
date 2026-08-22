"""Regenerate milestone-5.4 ideal shaped Grad-Shafranov measurement tables."""

from __future__ import annotations

import csv
import runpy
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one deterministic CSV table."""
    with path.open("w", newline="") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Execute the verification measurement helpers and replace both CSVs."""
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(root / "tests" / "verification" / "test_shaped_grad_shafranov.py")
    )
    zheng_rows: list[dict[str, Any]] = []
    measure_zheng = namespace["_zheng_row"]
    rate = namespace["_rate"]
    for polynomial_order in (1, 2, 3):
        measured = [measure_zheng(polynomial_order, maxh) for maxh in (0.20, 0.12, 0.07)]
        for index, row in enumerate(measured):
            values = {"polynomial_order": polynomial_order, **asdict(row)}
            if index + 1 < len(measured):
                finer = measured[index + 1]
                values["l2_rate_to_finer"] = rate(
                    row.relative_l2_error,
                    finer.relative_l2_error,
                    row.elements,
                    finer.elements,
                )
                values["energy_rate_to_finer"] = rate(
                    row.relative_weighted_energy_error,
                    finer.relative_weighted_energy_error,
                    row.elements,
                    finer.elements,
                )
            else:
                values["l2_rate_to_finer"] = ""
                values["energy_rate_to_finer"] = ""
            zheng_rows.append(values)
    _write(root / "tests" / "verification" / "shaped_zheng_rates.csv", zheng_rows)

    xpoint_rows = [asdict(row) for row in namespace["_cerfon_xpoint_rows"]()]
    for index, row in enumerate(xpoint_rows):
        if index + 1 < len(xpoint_rows):
            finer = xpoint_rows[index + 1]
            row["l2_rate_to_finer"] = rate(
                row["relative_l2_error"],
                finer["relative_l2_error"],
                row["elements"],
                finer["elements"],
            )
            row["geometry_rate_to_finer"] = rate(
                row["boundary_geometry_error"],
                finer["boundary_geometry_error"],
                row["elements"],
                finer["elements"],
            )
        else:
            row["l2_rate_to_finer"] = ""
            row["geometry_rate_to_finer"] = ""
    _write(root / "tests" / "verification" / "cerfon_freidberg_xpoint_rates.csv", xpoint_rows)


if __name__ == "__main__":
    main()
