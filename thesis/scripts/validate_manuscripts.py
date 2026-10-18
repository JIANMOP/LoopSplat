#!/usr/bin/env python3
"""Validate bilingual manuscript structure, evidence tokens, and local assets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from thesis.scripts.validate_bibliography import (
        citation_sequence,
        parse_bibliography,
    )
except ModuleNotFoundError:
    from validate_bibliography import citation_sequence, parse_bibliography


CITATION_RE = re.compile(r"@([A-Za-z0-9_:-]+)")
EQUATION_RE = re.compile(r"\\tag\{(\d+)\}")
LINK_RE = re.compile(r"\]\(([^)]+)\)")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|PLACEHOLDER)\b", re.IGNORECASE)
METRIC_RE = re.compile(r"\d[\d,]*\.\d+(?:%|°)?|\d+(?:\.\d+)?%")
PROHIBITED_PATTERNS = (
    (re.compile(r"state[- ]of[- ]the[- ]art", re.IGNORECASE), "state-of-the-art"),
    (re.compile(r"tightly coupled", re.IGNORECASE), "tightly coupled"),
    (
        re.compile(r"\bfirst\b[^\n.]{0,80}\b(?:IMU|inertial)\b", re.IGNORECASE),
        "unsupported first-use claim",
    ),
    (
        re.compile(
            r"\b(?:one|once|single|three|3)\b[^\n.]{0,50}\b(?:random\s+)?seeds?\b",
            re.IGNORECASE,
        ),
        "seed-count disclosure",
    ),
    (re.compile(r"紧耦合"), "紧耦合"),
    (re.compile(r"首次[^\n。]{0,60}(?:IMU|惯性)"), "unsupported first-use claim"),
    (
        re.compile(r"(?:单一|一个|一次|三组|三个|3个)[^\n。]{0,40}(?:随机|受控)?种子"),
        "seed-count disclosure",
    ),
)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _located_message(
    path: Path, text: str, description: str, needle: str | None = None
) -> str:
    offset = text.find(needle) if needle else 0
    if offset < 0:
        offset = 0
    return f"{path.name}:{_line_number(text, offset)}: {description}"


def _first_sequence_difference(left: list[str], right: list[str]) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _metric_numbers(text: str) -> list[str]:
    without_citations = re.sub(r"\[@[^]]+\]", "", text)
    return [
        match.group(0).replace(",", "")
        for match in METRIC_RE.finditer(without_citations)
    ]


def _number_set(text: str, english_label: str, chinese_label: str) -> set[int]:
    pattern = re.compile(
        rf"(?:\b{re.escape(english_label)}|{re.escape(chinese_label)})\s*(\d+)",
        re.IGNORECASE,
    )
    return {int(value) for value in pattern.findall(text)}


def _validate_contiguous_numbers(
    numbers: set[int], label: str, manuscript: Path, text: str
) -> list[str]:
    if not numbers:
        return [_located_message(manuscript, text, f"no {label} numbers found")]
    expected = set(range(1, max(numbers) + 1))
    if numbers != expected:
        return [
            _located_message(
                manuscript,
                text,
                f"non-contiguous {label} numbers: {sorted(numbers)}",
                str(min(numbers)),
            )
        ]
    return []


def _validate_single_manuscript(
    manuscript: Path, text: str, bibliography_keys: set[str]
) -> list[str]:
    errors: list[str] = []
    for match in PLACEHOLDER_RE.finditer(text):
        errors.append(
            f"{manuscript.name}:{_line_number(text, match.start())}: "
            f"unresolved placeholder {match.group(0)}"
        )
    for pattern, description in PROHIBITED_PATTERNS:
        for match in pattern.finditer(text):
            errors.append(
                f"{manuscript.name}:{_line_number(text, match.start())}: "
                f"prohibited unsupported claim: {description}"
            )

    citations = set(CITATION_RE.findall(text))
    for key in sorted(citations - bibliography_keys):
        errors.append(
            _located_message(
                manuscript,
                text,
                f"citation key absent from bibliography: {key}",
                f"@{key}",
            )
        )

    for match in LINK_RE.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "#")):
            continue
        asset = (manuscript.parent / target).resolve()
        if not asset.exists():
            errors.append(
                f"{manuscript.name}:{_line_number(text, match.start())}: "
                f"missing linked asset: {target}"
            )

    equation_tags = [int(value) for value in EQUATION_RE.findall(text)]
    if equation_tags:
        expected = list(range(1, max(equation_tags) + 1))
        if equation_tags != expected:
            errors.append(
                _located_message(
                    manuscript,
                    text,
                    "equation tags are not unique and contiguous: "
                    f"{equation_tags}",
                    rf"\tag{{{equation_tags[0]}}}",
                )
            )

    errors.extend(
        _validate_contiguous_numbers(
            _number_set(text, "Figure", "图"), "figure", manuscript, text
        )
    )
    errors.extend(
        _validate_contiguous_numbers(
            _number_set(text, "Table", "表"), "table", manuscript, text
        )
    )
    return errors


def validate_manuscripts(
    english_path: Path,
    chinese_path: Path,
    bibliography_path: Path,
    root: Path,
    require_citation_order: bool = False,
) -> list[str]:
    del root  # Link targets are resolved relative to each manuscript.
    paths = [Path(english_path), Path(chinese_path), Path(bibliography_path)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return [f"missing required file: {path}" for path in missing]

    english = Path(english_path).read_text(encoding="utf-8")
    chinese = Path(chinese_path).read_text(encoding="utf-8")
    bibliography_keys = {
        entry.key for entry in parse_bibliography(Path(bibliography_path))
    }

    errors = _validate_single_manuscript(
        Path(english_path), english, bibliography_keys
    )
    errors.extend(
        _validate_single_manuscript(
            Path(chinese_path), chinese, bibliography_keys
        )
    )

    english_citations = set(CITATION_RE.findall(english))
    chinese_citations = set(CITATION_RE.findall(chinese))
    if english_citations != chinese_citations:
        differing_key = sorted(english_citations ^ chinese_citations)[0]
        source_path, source_text = (
            (Path(english_path), english)
            if differing_key in english_citations
            else (Path(chinese_path), chinese)
        )
        errors.append(
            _located_message(
                source_path,
                source_text,
                f"bilingual citation-key sets differ at {differing_key}",
                f"@{differing_key}",
            )
        )

    if require_citation_order:
        english_sequence = citation_sequence(Path(english_path))
        chinese_sequence = citation_sequence(Path(chinese_path))
        if english_sequence != chinese_sequence:
            index = _first_sequence_difference(english_sequence, chinese_sequence)
            key = (
                english_sequence[index]
                if index < len(english_sequence)
                else chinese_sequence[index]
            )
            source_path, source_text = (
                (Path(english_path), english)
                if index < len(english_sequence)
                else (Path(chinese_path), chinese)
            )
            errors.append(
                _located_message(
                    source_path,
                    source_text,
                    f"bilingual first-citation sequences differ at {key}",
                    f"@{key}",
                )
            )

    english_equations = EQUATION_RE.findall(english)
    chinese_equations = EQUATION_RE.findall(chinese)
    if english_equations != chinese_equations:
        index = _first_sequence_difference(
            english_equations, chinese_equations
        )
        value = (
            english_equations[index]
            if index < len(english_equations)
            else chinese_equations[index]
        )
        source_path, source_text = (
            (Path(english_path), english)
            if index < len(english_equations)
            else (Path(chinese_path), chinese)
        )
        errors.append(
            _located_message(
                source_path,
                source_text,
                "bilingual equation-tag sequences differ",
                rf"\tag{{{value}}}",
            )
        )

    english_metrics = _metric_numbers(english)
    chinese_metrics = _metric_numbers(chinese)
    if english_metrics != chinese_metrics:
        index = _first_sequence_difference(english_metrics, chinese_metrics)
        value = (
            english_metrics[index]
            if index < len(english_metrics)
            else chinese_metrics[index]
        )
        source_path, source_text = (
            (Path(english_path), english)
            if index < len(english_metrics)
            else (Path(chinese_path), chinese)
        )
        errors.append(
            _located_message(
                source_path,
                source_text,
                "bilingual metric-number sequences differ",
                value,
            )
        )

    english_figures = _number_set(english, "Figure", "图")
    chinese_figures = _number_set(chinese, "Figure", "图")
    if english_figures != chinese_figures:
        value = sorted(english_figures ^ chinese_figures)[0]
        source_path, source_text, label = (
            (Path(english_path), english, "Figure")
            if value in english_figures
            else (Path(chinese_path), chinese, "图")
        )
        errors.append(
            _located_message(
                source_path,
                source_text,
                "bilingual figure-number sets differ",
                f"{label} {value}",
            )
        )

    english_tables = _number_set(english, "Table", "表")
    chinese_tables = _number_set(chinese, "Table", "表")
    if english_tables != chinese_tables:
        value = sorted(english_tables ^ chinese_tables)[0]
        source_path, source_text, label = (
            (Path(english_path), english, "Table")
            if value in english_tables
            else (Path(chinese_path), chinese, "表")
        )
        errors.append(
            _located_message(
                source_path,
                source_text,
                "bilingual table-number sets differ",
                f"{label} {value}",
            )
        )

    english_sections = re.findall(r"^##\s+(\d+)[.\s]", english, re.MULTILINE)
    chinese_sections = re.findall(r"^##\s+(\d+)[.\s]", chinese, re.MULTILINE)
    if english_sections != chinese_sections:
        errors.append(
            _located_message(
                Path(english_path),
                english,
                "bilingual numbered-section sequences differ",
            )
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("english", type=Path)
    parser.add_argument("chinese", type=Path)
    parser.add_argument("--bib", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--require-citation-order", action="store_true")
    args = parser.parse_args()
    errors = validate_manuscripts(
        args.english,
        args.chinese,
        args.bib,
        args.root,
        require_citation_order=args.require_citation_order,
    )
    if errors:
        for error in errors:
            print(error)
        return 1
    print("manuscript validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
