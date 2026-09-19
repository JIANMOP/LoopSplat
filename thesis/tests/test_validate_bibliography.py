from pathlib import Path

from thesis.scripts.validate_bibliography import validate_bibliography


def write(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_rejects_duplicate_keys(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(
        bibliography,
        """@article{dup,
  title = {First}, author = {Ada Lovelace}, year = {2024}
}
@article{dup,
  title = {Second}, author = {Grace Hopper}, year = {2024}
}
""",
    )

    errors = validate_bibliography(bibliography, [])

    assert any("duplicate citation key: dup" in error for error in errors)


def test_requires_title_author_and_year(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(bibliography, "@article{incomplete, title = {}, author = {}, year = {}}\n")

    errors = validate_bibliography(bibliography, [])

    assert "incomplete: missing title" in errors
    assert "incomplete: missing author" in errors
    assert "incomplete: missing year" in errors


def test_requires_preprint_marking_for_arxiv_misc_entries(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(
        bibliography,
        """@misc{unmarked,
  title = {Unmarked preprint},
  author = {Ada Lovelace},
  year = {2025},
  eprint = {2501.00001},
  archiveprefix = {arXiv}
}
""",
    )

    errors = validate_bibliography(bibliography, [])

    assert "unmarked: arXiv preprint must include keyword 'preprint'" in errors


def test_rejects_malformed_doi_values(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(
        bibliography,
        """@article{bad_doi,
  title = {Bad DOI},
  author = {Ada Lovelace},
  year = {2024},
  doi = {https://doi.org/10.1000/example}
}
""",
    )

    errors = validate_bibliography(bibliography, [])

    assert "bad_doi: malformed DOI: https://doi.org/10.1000/example" in errors


def test_reports_citations_missing_from_bibliography(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    manuscript = tmp_path / "manuscript-en.md"
    write(
        bibliography,
        """@article{present,
  title = {Present}, author = {Ada Lovelace}, year = {2024}
}
""",
    )
    write(manuscript, "Cite [@present] and [@missing].\n")

    errors = validate_bibliography(bibliography, [manuscript])

    assert "manuscript-en.md: citation key absent from bibliography: missing" in errors


def test_reports_entries_unused_by_every_manuscript(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    english = tmp_path / "manuscript-en.md"
    chinese = tmp_path / "manuscript-zh.md"
    write(
        bibliography,
        """@article{unused,
  title = {Unused}, author = {Ada Lovelace}, year = {2024}
}
@article{used,
  title = {Used}, author = {Grace Hopper}, year = {2024}
}
@article{candidate,
  title = {Candidate}, author = {Edsger Dijkstra}, year = {2024},
  keywords = {candidate}
}
""",
    )
    write(english, "English citation [@used].\n")
    write(chinese, "中文引文 [@used]。\n")

    errors = validate_bibliography(bibliography, [english, chinese])

    assert "unused: bibliography entry is unused by every manuscript" in errors
    assert "used: bibliography entry is unused by every manuscript" not in errors
    assert "candidate: bibliography entry is unused by every manuscript" not in errors
