#!/usr/bin/env python3
"""Build journal-neutral DOCX review copies from the manuscript masters."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


@dataclass(frozen=True)
class TableSpec:
    label: str
    target: str
    caption: str

    @property
    def link(self) -> str:
        return f"[{self.label}]({self.target})"


TABLE_SPECS = {
    "en": (
        TableSpec("Table 2", "tables/tum-results.md", "TUM RGB-D per-scene results."),
        TableSpec("Table 3", "tables/replica-results.md", "Replica per-scene results."),
        TableSpec("Table 4", "tables/fm-results.md", "FMDataset per-scene results."),
        TableSpec("Table 5", "tables/efficiency-results.md", "Paired efficiency measurements."),
    ),
    "zh": (
        TableSpec("表 2", "tables/tum-results.md", "TUM RGB-D 逐场景结果。"),
        TableSpec("表 3", "tables/replica-results.md", "Replica 逐场景结果。"),
        TableSpec("表 4", "tables/fm-results.md", "FMDataset 逐场景结果。"),
        TableSpec("表 5", "tables/efficiency-results.md", "配对效率测量。"),
    ),
}


@dataclass(frozen=True)
class ReviewInspection:
    table_count: int
    equation_count: int
    equation_numbers: tuple[int, ...]
    media_count: int
    drawing_count: int
    has_reference_heading: bool
    has_unresolved_citations: bool
    has_external_table_links: bool


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WORD = {"w": WORD_NS}
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NUMERIC_CSL = """<?xml version="1.0" encoding="utf-8"?>
<style xmlns="http://purl.org/net/xbiblio/csl" class="in-text" version="1.0">
  <info>
    <title>LoopSplat numeric review style</title>
    <id>https://example.invalid/styles/loopsplat-numeric-review</id>
    <updated>2026-10-18T00:00:00+00:00</updated>
  </info>
  <macro name="author">
    <names variable="author">
      <name initialize-with=". " and="text" et-al-min="7" et-al-use-first="6"/>
    </names>
  </macro>
  <macro name="title">
    <text variable="title"/>
  </macro>
  <macro name="container">
    <group delimiter=", ">
      <text variable="container-title" font-style="italic"/>
      <text variable="volume" font-weight="bold"/>
      <text variable="issue" prefix="(" suffix=")"/>
      <text variable="page"/>
    </group>
  </macro>
  <macro name="issued">
    <date variable="issued">
      <date-part name="year"/>
    </date>
  </macro>
  <citation collapse="citation-number">
    <sort>
      <key variable="citation-number"/>
    </sort>
    <layout prefix="[" suffix="]" delimiter=", ">
      <text variable="citation-number"/>
    </layout>
  </citation>
  <bibliography entry-spacing="0" second-field-align="flush">
    <layout suffix=".">
      <text variable="citation-number" prefix="[" suffix="]"/>
      <group delimiter=". ">
        <text macro="author"/>
        <text macro="title"/>
        <group delimiter=", ">
          <text macro="container"/>
          <text macro="issued"/>
        </group>
        <text variable="DOI" prefix="https://doi.org/"/>
      </group>
    </layout>
  </bibliography>
</style>
"""


def _word_tag(name: str) -> str:
    return f"{{{WORD_NS}}}{name}"


def assemble_review_markdown(
    manuscript: Path,
    thesis_root: Path,
    table_specs: tuple[TableSpec, ...],
    *,
    table_root: Path | None = None,
) -> str:
    """Return manuscript Markdown with linked result tables embedded once."""
    text = manuscript.read_text(encoding="utf-8")
    text = re.sub(r"\\\(([^\n]+?)\\\)", r"$\1$", text)
    text = re.sub(r"(?m)^\\\[$", "$$", text)
    text = re.sub(r"(?m)^\\\]$", "$$", text)
    text = re.sub(
        r"\\tag\{([^{}]+)\}",
        lambda match: rf"\qquad\text{{({match.group(1)})}}",
        text,
    )
    paragraphs = text.split("\n\n")
    for spec in reversed(table_specs):
        if table_root is None:
            link = spec.link
            source = thesis_root / spec.target
        else:
            source = table_root / Path(spec.target).name
            relative_target = Path(os.path.relpath(source, manuscript.parent)).as_posix()
            link = f"[{spec.label}]({relative_target})"
        matches = [index for index, paragraph in enumerate(paragraphs) if link in paragraph]
        if len(matches) != 1:
            raise ValueError(
                f"expected one manuscript link for {spec.label}, found {len(matches)}"
            )
        table = source.read_text(encoding="utf-8").strip()
        if not table.startswith("|"):
            raise ValueError(f"table source is not a Markdown pipe table: {source}")
        if source.name == "efficiency-results.md":
            table = table.replace("NVIDIA GeForce RTX ", "RTX ")
        index = matches[0]
        paragraphs[index] = paragraphs[index].replace(link, spec.label)
        paragraphs.insert(index + 1, f"**{spec.label}. {spec.caption}**\n\n{table}")
    return "\n\n".join(paragraphs).rstrip() + "\n"


def build_pandoc_command(
    pandoc: Path,
    source: Path,
    output: Path,
    bibliography: Path,
    thesis_root: Path,
    reference_heading: str,
    citation_style: Path | None = None,
) -> list[str]:
    """Construct the deterministic Pandoc command used for review copies."""
    command = [
        str(pandoc),
        str(source),
        "--from=markdown+citations+tex_math_dollars+pipe_tables",
        "--standalone",
        "--citeproc",
        "--fail-if-warnings",
        f"--bibliography={bibliography}",
        f"--resource-path={thesis_root}",
        "--metadata",
        f"reference-section-title={reference_heading}",
    ]
    if citation_style is not None:
        command.append(f"--csl={citation_style}")
    command.extend(("--output", str(output)))
    return command


def inspect_docx(
    document: Path,
    reference_heading: str,
    expected_equation_count: int = 11,
) -> ReviewInspection:
    """Inspect the Open XML payload without relying on a desktop application."""
    with ZipFile(document) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
        relationships = (
            archive.read("word/_rels/document.xml.rels").decode("utf-8")
            if "word/_rels/document.xml.rels" in archive.namelist()
            else ""
        )
        media_count = sum(
            name.startswith("word/media/") and not name.endswith("/")
            for name in archive.namelist()
        )
    root = ET.fromstring(xml)
    plain_text = "".join(root.itertext())
    display_equations = root.findall(f".//{{{MATH_NS}}}oMathPara")
    display_math_text = "".join(
        "".join(equation.itertext()) for equation in display_equations
    )
    equation_numbers = tuple(
        sorted(
            number for number in {
                int(value) for value in re.findall(r"\((\d+)\)", display_math_text)
            }
            if 1 <= number <= expected_equation_count
        )
    )
    return ReviewInspection(
        table_count=len(root.findall(".//w:tbl", WORD)),
        equation_count=len(display_equations),
        equation_numbers=equation_numbers,
        media_count=media_count,
        drawing_count=len(root.findall(".//w:drawing", WORD)),
        has_reference_heading=reference_heading in xml,
        has_unresolved_citations=bool(
            re.search(r"@[A-Za-z0-9_:-]+", plain_text)
        ),
        has_external_table_links="tables/" in xml + relationships,
    )


def validate_review_document(
    *,
    table_count: int,
    equation_count: int,
    equation_numbers: tuple[int, ...],
    media_count: int,
    drawing_count: int,
    has_reference_heading: bool,
    has_unresolved_citations: bool,
    has_external_table_links: bool,
    expected_equation_count: int = 11,
) -> None:
    """Reject an export that dropped required manuscript content."""
    errors = []
    if table_count < 5:
        errors.append(f"tables {table_count}/5")
    if equation_count < expected_equation_count:
        errors.append(f"equations {equation_count}/{expected_equation_count}")
    expected_equation_numbers = tuple(range(1, expected_equation_count + 1))
    if equation_numbers != expected_equation_numbers:
        if equation_numbers and equation_numbers == tuple(
            range(equation_numbers[0], equation_numbers[-1] + 1)
        ):
            found = f"{equation_numbers[0]}-{equation_numbers[-1]}"
        else:
            found = ",".join(map(str, equation_numbers)) or "none"
        errors.append(f"equation numbers {found}/1-{expected_equation_count}")
    if media_count < 8:
        errors.append(f"media {media_count}/8")
    if drawing_count < 8:
        errors.append(f"drawings {drawing_count}/8")
    if not has_reference_heading:
        errors.append("reference heading missing")
    if has_unresolved_citations:
        errors.append("unresolved citation keys")
    if has_external_table_links:
        errors.append("external table links")
    if errors:
        raise ValueError("incomplete review document: " + "; ".join(errors))


def format_review_docx(document: Path, language_tag: str) -> None:
    """Apply neutral A4 review formatting without changing manuscript sources."""
    ET.register_namespace("w", WORD_NS)
    with ZipFile(document) as source:
        xml = source.read("word/document.xml")
        root = ET.fromstring(xml)
        for run in root.findall(".//w:r", WORD):
            properties = run.find("w:rPr", WORD)
            if properties is None:
                properties = ET.Element(_word_tag("rPr"))
                run.insert(0, properties)
            language = properties.find("w:lang", WORD)
            if language is None:
                language = ET.SubElement(properties, _word_tag("lang"))
            language.set(_word_tag("val"), language_tag)
        for table in root.findall(".//w:tbl", WORD):
            if len(table.findall("w:tblGrid/w:gridCol", WORD)) >= 11:
                table_properties = table.find("w:tblPr", WORD)
                if table_properties is None:
                    table_properties = ET.Element(_word_tag("tblPr"))
                    table.insert(0, table_properties)
                layout = table_properties.find("w:tblLayout", WORD)
                if layout is None:
                    layout = ET.SubElement(table_properties, _word_tag("tblLayout"))
                layout.set(_word_tag("type"), "autofit")
                for row in table.findall("w:tr", WORD):
                    for column, cell in enumerate(row.findall("w:tc", WORD)):
                        if column >= 4:
                            continue
                        cell_properties = cell.find("w:tcPr", WORD)
                        if cell_properties is None:
                            cell_properties = ET.Element(_word_tag("tcPr"))
                            cell.insert(0, cell_properties)
                        if cell_properties.find("w:noWrap", WORD) is None:
                            ET.SubElement(cell_properties, _word_tag("noWrap"))
            for run in table.findall(".//w:r", WORD):
                properties = run.find("w:rPr", WORD)
                if properties is None:
                    properties = ET.Element(_word_tag("rPr"))
                    run.insert(0, properties)
                for name in ("sz", "szCs"):
                    size = properties.find(f"w:{name}", WORD)
                    if size is None:
                        size = ET.SubElement(properties, _word_tag(name))
                    size.set(_word_tag("val"), "16")
        for section in root.findall(".//w:sectPr", WORD):
            page_size = section.find("w:pgSz", WORD)
            if page_size is None:
                page_size = ET.SubElement(section, _word_tag("pgSz"))
            page_size.set(_word_tag("w"), "11906")
            page_size.set(_word_tag("h"), "16838")
            page_size.attrib.pop(_word_tag("orient"), None)
            margins = section.find("w:pgMar", WORD)
            if margins is None:
                margins = ET.SubElement(section, _word_tag("pgMar"))
            margins.set(_word_tag("top"), "1134")
            margins.set(_word_tag("bottom"), "1134")
            margins.set(_word_tag("left"), "850")
            margins.set(_word_tag("right"), "850")
        updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with tempfile.NamedTemporaryFile(
            prefix=document.stem + "-", suffix=".docx", dir=document.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with ZipFile(temporary_path, "w", ZIP_DEFLATED) as target:
                for member in source.infolist():
                    payload = (
                        updated_xml
                        if member.filename == "word/document.xml"
                        else source.read(member.filename)
                    )
                    target.writestr(member, payload)
            os.replace(temporary_path, document)
        finally:
            temporary_path.unlink(missing_ok=True)


def build_review_document(
    *,
    pandoc: Path,
    manuscript: Path,
    thesis_root: Path,
    output: Path,
    language: str,
    bibliography: Path | None = None,
    table_root: Path | None = None,
    numeric_citations: bool = False,
) -> ReviewInspection:
    """Assemble, export, and structurally verify one review document."""
    if language not in TABLE_SPECS:
        raise ValueError(f"unsupported language: {language}")
    if not pandoc.is_file():
        raise FileNotFoundError(f"Pandoc executable not found: {pandoc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_equation_count = len(
        set(re.findall(r"\\tag\{(\d+)\}", manuscript.read_text(encoding="utf-8")))
    )
    assembled = assemble_review_markdown(
        manuscript,
        thesis_root,
        TABLE_SPECS[language],
        table_root=table_root,
    )
    heading = "References" if language == "en" else "参考文献"
    language_tag = "en-US" if language == "en" else "zh-CN"
    with tempfile.TemporaryDirectory(prefix="loopsplat-review-") as temporary:
        source = Path(temporary) / f"assembled-{language}.md"
        source.write_text(assembled, encoding="utf-8")
        citation_style = None
        if numeric_citations:
            citation_style = Path(temporary) / "numeric.csl"
            citation_style.write_text(NUMERIC_CSL, encoding="utf-8")
        command = build_pandoc_command(
            pandoc,
            source,
            output,
            bibliography or thesis_root / "references.bib",
            thesis_root,
            heading,
            citation_style,
        )
        environment = os.environ.copy()
        environment["PATH"] = str(pandoc.parent) + os.pathsep + environment.get(
            "PATH", ""
        )
        subprocess.run(command, check=True, env=environment)
    format_review_docx(output, language_tag)
    inspection = inspect_docx(output, heading, expected_equation_count)
    validate_review_document(
        table_count=inspection.table_count,
        equation_count=inspection.equation_count,
        equation_numbers=inspection.equation_numbers,
        media_count=inspection.media_count,
        drawing_count=inspection.drawing_count,
        has_reference_heading=inspection.has_reference_heading,
        has_unresolved_citations=inspection.has_unresolved_citations,
        has_external_table_links=inspection.has_external_table_links,
        expected_equation_count=expected_equation_count,
    )
    return inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", type=Path, required=True)
    parser.add_argument("--thesis-root", type=Path, default=Path("thesis"))
    parser.add_argument("--manuscript-dir", type=Path)
    parser.add_argument("--bibliography", type=Path)
    parser.add_argument("--table-root", type=Path)
    parser.add_argument("--numeric-citations", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manuscript_dir = args.manuscript_dir or args.thesis_root
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.manuscript_dir is not None:
        output_dir = manuscript_dir / "review"
    else:
        output_dir = args.thesis_root / "review"
    for language in ("en", "zh"):
        output = output_dir / f"manuscript-{language}-review.docx"
        inspection = build_review_document(
            pandoc=args.pandoc,
            manuscript=manuscript_dir / f"manuscript-{language}.md",
            thesis_root=manuscript_dir,
            output=output,
            language=language,
            bibliography=args.bibliography,
            table_root=args.table_root,
            numeric_citations=args.numeric_citations,
        )
        print(
            f"built {output.name}: {inspection.table_count} tables, "
            f"{inspection.equation_count} display equations, "
            f"{inspection.media_count} media files"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
