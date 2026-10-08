#!/usr/bin/env python3

"""Download repository names from one hour of GH Archive data."""

import argparse
import gzip
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen


def archive_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD format") from error


def archive_hour(value: str) -> int:
    try:
        hour = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 23") from error

    if not 0 <= hour <= 23:
        raise argparse.ArgumentTypeError("must be from 0 to 23")
    return hour


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error

    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        dest="archive_date",
        type=archive_date,
        default=date(2015, 1, 1),
        metavar="YYYY-MM-DD",
        help="archive date (default: %(default)s)",
    )
    parser.add_argument(
        "--hour",
        type=archive_hour,
        default=15,
        help="archive hour from 0 to 23 (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        default=10_000,
        help="maximum number of names to write (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repos.txt"),
        metavar="FILE",
        help="output file (default: %(default)s)",
    )
    return parser.parse_args()


def download_repository_names(url: str, output: Path, limit: int) -> int:
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            request = Request(url, headers={"User-Agent": "UPGen archive downloader"})

            with urlopen(request) as response:
                with gzip.GzipFile(fileobj=response) as archive:
                    count = 0
                    for line in archive:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            continue

                        repository = event.get("repo")
                        if not isinstance(repository, dict):
                            continue

                        name = repository.get("name")
                        if not isinstance(name, str) or not name:
                            continue

                        destination.write(f"{name}\n")
                        count += 1
                        if count == limit:
                            break

        os.replace(temporary_path, output)
        temporary_path = None
        return count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    if not args.output.name:
        raise SystemExit("error: --output must name a file")
    if not args.output.parent.is_dir():
        message = f"error: output directory does not exist: {args.output.parent}"
        raise SystemExit(message)
    if args.output.is_dir():
        raise SystemExit(f"error: output path is a directory: {args.output}")

    url = (
        "https://data.gharchive.org/"
        f"{args.archive_date.isoformat()}-{args.hour}.json.gz"
    )
    print(f"Downloading {url}")

    try:
        count = download_repository_names(url, args.output, args.limit)
    except (OSError, EOFError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(f"Wrote {count} repository names to {args.output}")


if __name__ == "__main__":
    main()
