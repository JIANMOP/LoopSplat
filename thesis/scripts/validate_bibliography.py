#!/usr/bin/env python3
"""Validate the paper's BibTeX integrity and Markdown citation coverage."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
CITATION_RE = re.compile(r"@([A-Za-z0-9_:-]+)")


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]


def _balanced_segment(text: str, start: int, opening: str, closing: str) -> tuple[str, int]:
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError("unclosed BibTeX delimiter")


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    position = 0
    while position < len(body):
        match = re.search(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", body[position:])
        if match is None:
            break
        name = match.group(1).lower()
        value_start = position + match.end()
        while value_start < len(body) and body[value_start].isspace():
            value_start += 1
        if value_start >= len(body):
            fields[name] = ""
            break
        if body[value_start] == "{":
            value, position = _balanced_segment(body, value_start, "{", "}")
        elif body[value_start] == '"':
            end = body.find('"', value_start + 1)
            if end < 0:
                raise ValueError("unclosed quoted BibTeX value")
            value, position = body[value_start + 1 : end], end + 1
        else:
            end = body.find(",", value_start)
            if end < 0:
                end = len(body)
            value, position = body[value_start:end].strip(), end
        fields[name] = value.strip()
    return fields


def parse_bibliography(path: Path) -> list[BibEntry]:
    text = Path(path).read_text(encoding="utf-8")
    entries: list[BibEntry] = []
    position = 0
    header = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
    while True:
        match = header.search(text, position)
        if match is None:
            break
        brace_start = text.find("{", match.start(), match.end())
        body, position = _balanced_segment(text, brace_start, "{", "}")
        entries.append(
            BibEntry(
                entry_type=match.group(1).lower(),
                key=match.group(2),
                fields=_parse_fields(body),
            )
        )
    return entries


def validate_bibliography(bib_path: Path, manuscripts: list[Path]) -> list[str]:
    errors: list[str] = []
    try:
        entries = parse_bibliography(bib_path)
    except ValueError as exc:
        return [str(exc)]
    seen: set[str] = set()
    for entry in entries:
        if entry.key in seen:
            errors.append(f"duplicate citation key: {entry.key}")
        seen.add(entry.key)
        for required in ("title", "author", "year"):
            if not entry.fields.get(required, "").strip():
                errors.append(f"{entry.key}: missing {required}")
        doi = entry.fields.get("doi")
        if doi and not DOI_RE.fullmatch(doi.strip()):
            errors.append(f"{entry.key}: malformed DOI: {doi}")
        url = entry.fields.get("url")
        if url and not URL_RE.fullmatch(url.strip()):
            errors.append(f"{entry.key}: malformed URL: {url}")
        is_arxiv = entry.entry_type == "misc" and (
            entry.fields.get("archiveprefix", "").lower() == "arxiv"
            or "arxiv" in entry.fields.get("howpublished", "").lower()
        )
        if is_arxiv and "preprint" not in entry.fields.get("keywords", "").lower():
            errors.append(
                f"{entry.key}: arXiv preprint must include keyword 'preprint'"
            )

    if manuscripts:
        cited: set[str] = set()
        for manuscript in manuscripts:
            text = Path(manuscript).read_text(encoding="utf-8")
            keys = set(CITATION_RE.findall(text))
            cited.update(keys)
            for key in sorted(keys - seen):
                errors.append(
                    f"{Path(manuscript).name}: citation key absent from bibliography: {key}"
                )
        for entry in entries:
            is_candidate = "candidate" in entry.fields.get("keywords", "").lower()
            if entry.key not in cited and not is_candidate:
                errors.append(
                    f"{entry.key}: bibliography entry is unused by every manuscript"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bibliography", type=Path)
    parser.add_argument("manuscripts", nargs="*", type=Path)
    args = parser.parse_args()
    errors = validate_bibliography(args.bibliography, args.manuscripts)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("bibliography validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
