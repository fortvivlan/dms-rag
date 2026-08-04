"""Split codex.csv rows into legal parts and numbered points."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator


FIELDNAMES = ("text", "source")

# A point number is structural when it starts the text or follows punctuation
# that can separate list items. This deliberately excludes references such as
# ``(статья 426)`` and amendment notes such as ``N 273)``.
POINT_MARKER_RE = re.compile(
    r"(?:^\s*|[;:.!?)]\s+)(?P<number>\d+(?:\.\d+)*)\)(?=\s)"
)

# The existing CSV incorrectly calls article parts "п." in source metadata.
PART_SOURCE_RE = re.compile(
    r"^(?P<base>.+?)\s+п\.\s+(?P<number>\d+(?:\.\d+)*)\.\s*$"
)


def split_text_into_points(text: str) -> list[tuple[str, str]]:
    """Return ``(point_number, text)`` rows, duplicating the shared preamble."""
    markers = list(POINT_MARKER_RE.finditer(text))
    if not markers:
        return []

    preamble = text[: markers[0].start("number")].strip()
    points: list[tuple[str, str]] = []

    for index, marker in enumerate(markers):
        start = marker.start("number")
        end = (
            markers[index + 1].start("number")
            if index + 1 < len(markers)
            else len(text)
        )
        point_text = text[start:end].strip()
        combined_text = " ".join(part for part in (preamble, point_text) if part)
        points.append((marker.group("number"), combined_text))

    return points


def split_source(source: str) -> tuple[str, str | None]:
    """Return the article-level source and its existing part number, if any."""
    match = PART_SOURCE_RE.fullmatch(source)
    if match is None:
        return source.strip(), None
    return match.group("base").rstrip(), match.group("number")


def format_source(base: str, part_number: str | None, point_number: str | None) -> str:
    """Build corrected source metadata for an article, part, or point."""
    suffixes: list[str] = []
    if part_number is not None:
        suffixes.append(f"ч. {part_number}")
    if point_number is not None:
        suffixes.append(f"п. {point_number}")
    if not suffixes:
        return base
    return f"{base} {' '.join(suffixes)}."


def reparse_rows(rows: Iterator[dict[str, str]]) -> Iterator[dict[str, str]]:
    """Yield rows split by points with corrected source metadata."""
    for row_number, row in enumerate(rows, start=2):
        if tuple(row) != FIELDNAMES:
            raise ValueError(
                f"Unexpected columns in CSV row {row_number}: {list(row)}; "
                f"expected {list(FIELDNAMES)}"
            )

        text = row["text"]
        source = row["source"]
        if not text.strip() or not source.strip():
            raise ValueError(f"Empty text or source in CSV row {row_number}")

        base_source, part_number = split_source(source)
        points = split_text_into_points(text)

        if not points:
            yield {
                "text": text,
                "source": format_source(base_source, part_number, None),
            }
            continue

        for point_number, point_text in points:
            yield {
                "text": point_text,
                "source": format_source(base_source, part_number, point_number),
            }


def reparse_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Reparse *input_path* into *output_path* and return input/output counts."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with input_path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames != list(FIELDNAMES):
                raise ValueError(
                    f"Unexpected CSV header: {reader.fieldnames}; "
                    f"expected {list(FIELDNAMES)}"
                )

            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
                writer.writeheader()

                input_count = 0
                output_count = 0

                def counted_rows() -> Iterator[dict[str, str]]:
                    nonlocal input_count
                    for input_row in reader:
                        input_count += 1
                        yield input_row

                for result_row in reparse_rows(counted_rows()):
                    writer.writerow(result_row)
                    output_count += 1

        os.replace(temporary_path, output_path)
        temporary_path = None
        return input_count, output_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split codex CSV rows into parts and numbered points."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("codex.csv"),
        help="source CSV (default: codex.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codex_parts.csv"),
        help="destination CSV (default: codex_parts.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_count, output_count = reparse_csv(args.input, args.output)
    print(
        f"Reparsed {input_count} rows into {output_count} rows: {args.output}"
    )


if __name__ == "__main__":
    main()
