from pathlib import Path
import re
import subprocess
import sys

from thesis.scripts.validate_manuscripts import validate_manuscripts


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def bibliography(path: Path) -> None:
    write(
        path,
        """@article{source,
  title = {Verified source},
  author = {Ada Lovelace},
  year = {2024},
  url = {https://example.org/source}
}
""",
    )


def valid_pair(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "thesis"
    english = root / "manuscript-en.md"
    chinese = root / "manuscript-zh.md"
    bib = root / "references.bib"
    write(root / "figures" / "overview.svg", "<svg/>")
    write(
        english,
        """# Paper
## 1. Method
See Figure 1 and Table 1 [@source].
![Figure 1. Overview.](figures/overview.svg)
**Table 1. Result.**
The score is 1.250 dB.
$$x=1.\\tag{1}$$
""",
    )
    write(
        chinese,
        """# 论文
## 1 方法
见图 1 和表 1 [@source]。
![图 1 总体结构。](figures/overview.svg)
**表 1 结果。**
该指标为 1.250 dB。
$$x=1.\\tag{1}$$
""",
    )
    bibliography(bib)
    return english, chinese, bib, root


def test_accepts_a_consistent_bilingual_pair(tmp_path: Path) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)

    assert validate_manuscripts(english, chinese, bib, root) == []


def test_cli_runs_as_a_direct_script(tmp_path: Path) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)
    script = Path(__file__).parents[1] / "scripts" / "validate_manuscripts.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(english),
            str(chinese),
            "--bib",
            str(bib),
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "manuscript validation passed" in completed.stdout


def test_reports_placeholders_and_prohibited_claims(tmp_path: Path) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)
    english.write_text(
        english.read_text(encoding="utf-8")
        + "\nTODO: claim state-of-the-art performance under one controlled random seed.\n",
        encoding="utf-8",
    )

    errors = validate_manuscripts(english, chinese, bib, root)

    assert any("TODO" in error and "manuscript-en.md" in error for error in errors)
    assert any("state-of-the-art" in error for error in errors)
    assert any("seed-count disclosure" in error for error in errors)


def test_reports_missing_assets_and_unresolved_citations(tmp_path: Path) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)
    english.write_text(
        english.read_text(encoding="utf-8")
        .replace("figures/overview.svg", "figures/missing.svg")
        .replace("[@source]", "[@missing]"),
        encoding="utf-8",
    )

    errors = validate_manuscripts(english, chinese, bib, root)

    assert any("missing linked asset" in error for error in errors)
    assert any(
        re.search(r"manuscript-en\.md:\d+:", error)
        and "citation key absent from bibliography: missing" in error
        for error in errors
    )


def test_reports_bilingual_citation_and_equation_mismatches(tmp_path: Path) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)
    chinese.write_text(
        chinese.read_text(encoding="utf-8")
        .replace("[@source]", "")
        .replace(r"\tag{1}", r"\tag{2}"),
        encoding="utf-8",
    )

    errors = validate_manuscripts(english, chinese, bib, root)

    assert any(
        "bilingual citation-key sets differ" in error
        and "manuscript-en.md:" in error
        for error in errors
    )
    assert any(
        "bilingual equation-tag sequences differ" in error
        and "manuscript-en.md:" in error
        for error in errors
    )


def test_reports_line_numbers_for_noncontiguous_local_numbering(
    tmp_path: Path,
) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)
    english.write_text(
        english.read_text(encoding="utf-8")
        .replace(r"\tag{1}", r"\tag{2}")
        .replace("Figure 1", "Figure 2")
        .replace("Table 1", "Table 2"),
        encoding="utf-8",
    )

    errors = validate_manuscripts(english, chinese, bib, root)

    assert any(
        re.search(r"manuscript-en\.md:\d+:", error)
        and "equation tags are not unique and contiguous" in error
        for error in errors
    )
    assert any(
        re.search(r"manuscript-en\.md:\d+:", error)
        and "non-contiguous figure numbers" in error
        for error in errors
    )
    assert any(
        re.search(r"manuscript-en\.md:\d+:", error)
        and "non-contiguous table numbers" in error
        for error in errors
    )


def test_reports_metric_and_figure_table_mismatches(tmp_path: Path) -> None:
    english, chinese, bib, root = valid_pair(tmp_path)
    chinese.write_text(
        chinese.read_text(encoding="utf-8")
        .replace("1.250 dB", "1.500 dB")
        .replace("图 1", "图 2")
        .replace("表 1", "表 2"),
        encoding="utf-8",
    )

    errors = validate_manuscripts(english, chinese, bib, root)

    assert any(
        "bilingual metric-number sequences differ" in error
        and "manuscript-en.md:" in error
        for error in errors
    )
    assert any(
        "bilingual figure-number sets differ" in error
        and "manuscript-en.md:" in error
        for error in errors
    )
    assert any(
        "bilingual table-number sets differ" in error
        and "manuscript-en.md:" in error
        for error in errors
    )
