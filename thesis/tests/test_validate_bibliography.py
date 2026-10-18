from pathlib import Path

from thesis.scripts import validate_bibliography as bibliography_validator
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


def test_rejects_malformed_reference_urls(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(
        bibliography,
        """@article{bad_url,
  title = {Bad URL},
  author = {Ada Lovelace},
  year = {2024},
  url = {doi.org/10.1000/example}
}
""",
    )

    errors = validate_bibliography(bibliography, [])

    assert "bad_url: malformed URL: doi.org/10.1000/example" in errors


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


def test_returns_unique_citations_in_first_appearance_order(tmp_path: Path) -> None:
    manuscript = tmp_path / "manuscript-en.md"
    write(manuscript, "See [@beta; @alpha] and again [@beta].\n")

    assert bibliography_validator.citation_sequence(manuscript) == ["beta", "alpha"]


def test_strict_mode_rejects_arxiv_and_misc_entries(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(
        bibliography,
        """@misc{misc_preprint,
  title = {Misc preprint}, author = {Ada Lovelace}, year = {2025},
  doi = {10.1000/misc}, archiveprefix = {arXiv}
}
@article{eprint_preprint,
  title = {Eprint preprint}, author = {Ada Lovelace}, year = {2025},
  doi = {10.1000/eprint}, eprint = {2501.00001}
}
@article{archive_preprint,
  title = {Archive preprint}, author = {Ada Lovelace}, year = {2025},
  doi = {10.1000/archive}, archiveprefix = {arXiv}
}
""",
    )

    errors = validate_bibliography(bibliography, [], published_only=True)

    assert "misc_preprint: unpublished entry type: misc" in errors
    assert "eprint_preprint: arXiv preprint metadata is not allowed" in errors
    assert "archive_preprint: arXiv preprint metadata is not allowed" in errors


def test_strict_mode_requires_a_bare_doi(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    write(
        bibliography,
        """@article{missing_doi,
  title = {Missing DOI}, author = {Ada Lovelace}, year = {2024}
}
@article{prefixed_doi,
  title = {Prefixed DOI}, author = {Grace Hopper}, year = {2024},
  doi = {https://doi.org/10.1000/prefixed}
}
""",
    )

    errors = validate_bibliography(bibliography, [], require_doi=True)

    assert "missing_doi: missing DOI" in errors
    assert "prefixed_doi: malformed DOI: https://doi.org/10.1000/prefixed" in errors


def test_strict_mode_rejects_uncited_entries(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    manuscript = tmp_path / "manuscript-en.md"
    write(
        bibliography,
        """@article{cited,
  title = {Cited}, author = {Ada Lovelace}, year = {2024}
}
@article{uncited,
  title = {Uncited}, author = {Grace Hopper}, year = {2024},
  keywords = {candidate}
}
""",
    )
    write(manuscript, "See [@cited].\n")

    errors = validate_bibliography(bibliography, [manuscript], require_order=True)

    assert "uncited: bibliography entry is unused by every manuscript" in errors


def test_strict_mode_requires_bibtex_order_to_match_first_citations(
    tmp_path: Path,
) -> None:
    bibliography = tmp_path / "references.bib"
    manuscript = tmp_path / "manuscript-en.md"
    write(
        bibliography,
        """@article{alpha,
  title = {Alpha}, author = {Ada Lovelace}, year = {2024}
}
@article{beta,
  title = {Beta}, author = {Grace Hopper}, year = {2024}
}
""",
    )
    write(manuscript, "See [@beta] before [@alpha].\n")

    errors = validate_bibliography(bibliography, [manuscript], require_order=True)

    assert any(
        "BibTeX entry sequence differs from first citation sequence at beta" in error
        for error in errors
    )
