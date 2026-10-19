from pathlib import Path
from shutil import copytree, ignore_patterns
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from thesis.scripts.build_review_documents import (
    TABLE_SPECS,
    assemble_review_markdown,
    build_review_document,
    build_pandoc_command,
    format_review_docx,
    inspect_docx,
    validate_review_document,
)


def test_assemble_review_markdown_inlines_tables_without_editing_sources(
    tmp_path: Path,
) -> None:
    thesis_root = tmp_path / "thesis"
    tables = thesis_root / "tables"
    tables.mkdir(parents=True)
    manuscript = thesis_root / "manuscript-en.md"
    manuscript.write_text(
        "# Paper\n\nFor \\(x_i\\), define\n\n$$\nx=1.\n\\tag{1}\n$$\n\n"
        "Results are listed in [Table 2](tables/tum-results.md).\n",
        encoding="utf-8",
    )
    table = tables / "tum-results.md"
    table.write_text("| Scene | PSNR |\n|---|---:|\n| A1 | 12.3 |\n", encoding="utf-8")

    assembled = assemble_review_markdown(
        manuscript,
        thesis_root,
        (TABLE_SPECS["en"][0],),
    )

    assert "[Table 2](tables/tum-results.md)" not in assembled
    assert "Results are listed in Table 2." in assembled
    assert "**Table 2. TUM RGB-D per-scene results.**" in assembled
    assert assembled.count("| A1 | 12.3 |") == 1
    assert r"\tag{1}" not in assembled
    assert r"\qquad\text{(1)}" in assembled
    assert r"For $x_i$, define" in assembled
    assert r"\tag{1}" in manuscript.read_text(encoding="utf-8")
    assert r"For \(x_i\), define" in manuscript.read_text(encoding="utf-8")


def test_assemble_review_markdown_compacts_gpu_names_only_in_efficiency_export(
    tmp_path: Path,
) -> None:
    thesis_root = tmp_path / "thesis"
    tables = thesis_root / "tables"
    tables.mkdir(parents=True)
    manuscript = thesis_root / "manuscript-en.md"
    manuscript.write_text(
        "# Paper\n\nSee [Table 5](tables/efficiency-results.md).\n",
        encoding="utf-8",
    )
    table = tables / "efficiency-results.md"
    table.write_text(
        "| GPU | Time |\n|---|---:|\n"
        "| NVIDIA GeForce RTX 2080 Ti | 1.0 |\n"
        "| NVIDIA GeForce RTX 4080 SUPER | 2.0 |\n",
        encoding="utf-8",
    )

    assembled = assemble_review_markdown(
        manuscript,
        thesis_root,
        (TABLE_SPECS["en"][3],),
    )

    assert "| RTX 2080 Ti | 1.0 |" in assembled
    assert "| RTX 4080 SUPER | 2.0 |" in assembled
    assert "NVIDIA GeForce" not in assembled
    assert "NVIDIA GeForce" in table.read_text(encoding="utf-8")


def test_assemble_review_markdown_uses_explicit_shared_table_root_for_parent_links(
    tmp_path: Path,
) -> None:
    manuscript_dir = tmp_path / "thesis" / "v2"
    table_root = tmp_path / "thesis" / "tables"
    manuscript_dir.mkdir(parents=True)
    table_root.mkdir(parents=True)
    manuscript = manuscript_dir / "manuscript-en.md"
    manuscript.write_text(
        "# Paper\n\nResults are listed in [Table 2](../tables/tum-results.md).\n",
        encoding="utf-8",
    )
    table = table_root / "tum-results.md"
    table.write_text("| Scene | PSNR |\n|---|---:|\n| A1 | 12.3 |\n", encoding="utf-8")

    assembled = assemble_review_markdown(
        manuscript,
        manuscript_dir,
        (TABLE_SPECS["en"][0],),
        table_root=table_root,
    )

    assert "[Table 2](../tables/tum-results.md)" not in assembled
    assert "Results are listed in Table 2." in assembled
    assert assembled.count("| A1 | 12.3 |") == 1
    assert "[Table 2](../tables/tum-results.md)" in manuscript.read_text(
        encoding="utf-8"
    )


def test_build_pandoc_command_enables_citations_math_and_local_assets(
    tmp_path: Path,
) -> None:
    thesis_root = tmp_path / "thesis"
    source = thesis_root / "assembled.md"
    output = thesis_root / "review" / "paper.docx"
    bibliography = thesis_root / "references.bib"

    command = build_pandoc_command(
        Path("/usr/bin/pandoc"),
        source,
        output,
        bibliography,
        thesis_root,
        "References",
    )

    assert command[0] == "/usr/bin/pandoc"
    assert "--citeproc" in command
    assert "--fail-if-warnings" in command
    assert f"--bibliography={bibliography}" in command
    assert f"--resource-path={thesis_root}" in command
    assert "--from=markdown+citations+tex_math_dollars+pipe_tables" in command
    assert "reference-section-title=References" in command
    assert command[-2:] == ["--output", str(output)]


def test_inspect_docx_counts_tables_equations_media_and_references(
    tmp_path: Path,
) -> None:
    document = tmp_path / "review.docx"
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><w:body>'
        '<w:tbl/><w:tbl/><m:oMathPara><m:oMath><m:r><m:t>(1)</m:t></m:r>'
        '</m:oMath></m:oMathPara>'
        '<w:p><m:oMath><m:r><m:t>(2)</m:t></m:r></m:oMath></w:p>'
        '<w:p><w:r><w:t>References</w:t>'
        "</w:r></w:p></w:body></w:document>"
    )
    with ZipFile(document, "w") as archive:
        archive.writestr("word/document.xml", xml)
        archive.writestr("word/media/image1.png", b"one")
        archive.writestr("word/media/image2.jpg", b"two")

    inspection = inspect_docx(document, "References")

    assert inspection.table_count == 2
    assert inspection.equation_count == 1
    assert inspection.media_count == 2
    assert inspection.drawing_count == 0
    assert inspection.has_reference_heading
    assert inspection.equation_numbers == (1,)
    assert not inspection.has_unresolved_citations
    assert not inspection.has_external_table_links


def test_validate_review_document_rejects_incomplete_exports() -> None:
    try:
        validate_review_document(
            table_count=4,
            equation_count=10,
            equation_numbers=tuple(range(1, 11)),
            media_count=7,
            drawing_count=7,
            has_reference_heading=False,
            has_unresolved_citations=True,
            has_external_table_links=True,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("incomplete review document was accepted")

    assert "tables 4/5" in message
    assert "equations 10/11" in message
    assert "equation numbers 1-10/1-11" in message
    assert "media 7/8" in message
    assert "drawings 7/8" in message
    assert "reference heading missing" in message
    assert "unresolved citation keys" in message
    assert "external table links" in message


def test_validate_review_document_accepts_all_twenty_v2_equations() -> None:
    validate_review_document(
        table_count=5,
        equation_count=20,
        equation_numbers=tuple(range(1, 21)),
        media_count=8,
        drawing_count=8,
        has_reference_heading=True,
        has_unresolved_citations=False,
        has_external_table_links=False,
        expected_equation_count=20,
    )


def test_build_review_document_exports_the_complete_english_master(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    thesis_root = repository / "thesis"
    output = tmp_path / "manuscript-en-review.docx"
    pandoc = Path(sys.executable).with_name("pandoc")

    inspection = build_review_document(
        pandoc=pandoc,
        manuscript=thesis_root / "manuscript-en.md",
        thesis_root=thesis_root,
        output=output,
        language="en",
    )

    assert output.is_file()
    assert output.stat().st_size > 100_000
    assert inspection.table_count >= 5
    assert inspection.equation_count >= 11
    assert inspection.media_count >= 8
    assert inspection.drawing_count >= 8
    assert inspection.has_reference_heading
    assert not inspection.has_unresolved_citations
    assert not inspection.has_external_table_links


def test_cli_builds_bilingual_review_documents(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    thesis_root = repository / "thesis"
    script = thesis_root / "scripts" / "build_review_documents.py"
    pandoc = Path(sys.executable).with_name("pandoc")
    output_dir = tmp_path / "review"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pandoc",
            str(pandoc),
            "--thesis-root",
            str(thesis_root),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "manuscript-en-review.docx").stat().st_size > 100_000
    assert (output_dir / "manuscript-zh-review.docx").stat().st_size > 100_000
    assert "built manuscript-en-review.docx" in completed.stdout
    assert "built manuscript-zh-review.docx" in completed.stdout
    assert "display equations" in completed.stdout
    assert "WARNING" not in completed.stderr


def test_cli_builds_v2_from_explicit_inputs_without_touching_v1_reviews(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    thesis_root = repository / "thesis"
    manuscript_dir = thesis_root / "v2"
    script = thesis_root / "scripts" / "build_review_documents.py"
    pandoc = Path(sys.executable).with_name("pandoc")
    output_dir = tmp_path / "review"
    bibliography = tmp_path / "references.bib"
    bibliography.write_text(
        (manuscript_dir / "references.bib")
        .read_text(encoding="utf-8")
        .replace(
            "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            "Explicit Bibliography Path Sentinel",
        ),
        encoding="utf-8",
    )
    v1_reviews = tuple(sorted((thesis_root / "review").glob("manuscript-*-review.*")))
    before = {document: document.read_bytes() for document in v1_reviews}

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pandoc",
            str(pandoc),
            "--manuscript-dir",
            str(manuscript_dir),
            "--bibliography",
            str(bibliography),
            "--table-root",
            str(thesis_root / "tables"),
            "--numeric-citations",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for language, heading in (("en", "References"), ("zh", "参考文献")):
        document = output_dir / f"manuscript-{language}-review.docx"
        assert document.stat().st_size > 100_000
        inspection = inspect_docx(
            document,
            heading,
            expected_equation_count=20,
        )
        validate_review_document(
            table_count=inspection.table_count,
            equation_count=inspection.equation_count,
            equation_numbers=inspection.equation_numbers,
            media_count=inspection.media_count,
            drawing_count=inspection.drawing_count,
            has_reference_heading=inspection.has_reference_heading,
            has_unresolved_citations=inspection.has_unresolved_citations,
            has_external_table_links=inspection.has_external_table_links,
            expected_equation_count=20,
        )
        with ZipFile(document) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        plain_text = "".join(root.itertext())
        assert "[1]" in plain_text
        assert "(Kerbl" not in plain_text
        assert "Explicit bibliography path sentinel" in plain_text
    assert {document: document.read_bytes() for document in v1_reviews} == before


def test_cli_defaults_explicit_v2_inputs_to_its_review_directory(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    source_thesis = repository / "thesis"
    thesis_root = tmp_path / "thesis"
    manuscript_dir = thesis_root / "v2"
    copytree(
        source_thesis / "v2",
        manuscript_dir,
        ignore=ignore_patterns("review"),
    )
    copytree(source_thesis / "tables", thesis_root / "tables")
    copytree(source_thesis / "figures", thesis_root / "figures")
    script = source_thesis / "scripts" / "build_review_documents.py"
    pandoc = Path(sys.executable).with_name("pandoc")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pandoc",
            str(pandoc),
            "--manuscript-dir",
            str(manuscript_dir),
            "--bibliography",
            str(manuscript_dir / "references.bib"),
            "--table-root",
            str(thesis_root / "tables"),
            "--numeric-citations",
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    review_dir = manuscript_dir / "review"
    assert (review_dir / "manuscript-en-review.docx").is_file()
    assert (review_dir / "manuscript-zh-review.docx").is_file()
    assert not (thesis_root / "review").exists()


def test_cli_defaults_legacy_inputs_to_the_thesis_root_review_directory(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    source_thesis = repository / "thesis"
    thesis_root = tmp_path / "isolated-thesis"
    thesis_root.mkdir()
    for name in ("manuscript-en.md", "manuscript-zh.md", "references.bib"):
        (thesis_root / name).write_bytes((source_thesis / name).read_bytes())
    copytree(source_thesis / "tables", thesis_root / "tables")
    copytree(source_thesis / "figures", thesis_root / "figures")
    script = source_thesis / "scripts" / "build_review_documents.py"
    pandoc = Path(sys.executable).with_name("pandoc")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pandoc",
            str(pandoc),
            "--thesis-root",
            str(thesis_root),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    review_dir = thesis_root / "review"
    assert (review_dir / "manuscript-en-review.docx").is_file()
    assert (review_dir / "manuscript-zh-review.docx").is_file()


def test_format_review_docx_sets_a4_margins_and_compact_table_text(
    tmp_path: Path,
) -> None:
    document = tmp_path / "review.docx"
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Body text</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table text</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    with ZipFile(document, "w") as archive:
        archive.writestr("word/document.xml", xml)
        archive.writestr("word/styles.xml", "styles")

    format_review_docx(document, "zh-CN")

    with ZipFile(document) as archive:
        updated = archive.read("word/document.xml").decode("utf-8")
        assert archive.read("word/styles.xml") == b"styles"
    assert 'w:w="11906"' in updated
    assert 'w:h="16838"' in updated
    assert 'w:left="850"' in updated
    assert 'w:right="850"' in updated
    assert 'w:val="16"' in updated
    assert updated.count('w:val="16"') == 2
    assert 'w:lang w:val="zh-CN"' in updated
