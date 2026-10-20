# OARS-GS v2 final-fix report

## Status

Resolved the final-review findings without changing datasets, recorded results, source SLAM/configuration files, bibliography, opening report, or v1 material.

Commit: `docs: resolve OARS-GS final review findings`.

## Changes

- `thesis/scripts/build_review_documents.py` now inserts same-paragraph result tables in source order. Its DOCX formatter uses an auto-fit layout and no-wrap labels only for the eleven-column efficiency table.
- `thesis/tests/test_build_review_documents.py` covers ordered same-paragraph table insertion, bilingual DOCX figure numbering/order, and the narrow-table versus wide-table formatting boundary.
- `thesis/v2/manuscript-en.md`, `thesis/v2/manuscript-zh.md`, and their method/results section sources explicitly number Figures 1--8; correct the Eq. (2) mask description; and place optional final loop closure before checkpointing with an inactive-config note.
- `thesis/v2/internal/method-code-audit.md` records the depth/RGB mask distinction and final loop-closure order.
- `scripts/run_ablation.py` and `tests/test_experiment_runner.py` label the default three-seed path as an optional robustness extension and distinguish it from the selected seed-0 manuscript source.
- Regenerated review artifacts: `thesis/v2/review/manuscript-en-review.docx`, `thesis/v2/review/manuscript-en-review.pdf`, `thesis/v2/review/manuscript-zh-review.docx`, and `thesis/v2/review/manuscript-zh-review.pdf`.

## TDD evidence

The table-insertion and initial narrow-cell formatting fixes were recovered from the dirty worktree as untrusted material. Their regression tests were deliberately run against the restored base builder behavior:

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  thesis/tests/test_build_review_documents.py::test_assemble_review_markdown_preserves_table_order_for_links_in_one_paragraph \
  thesis/tests/test_build_review_documents.py::test_format_review_docx_sets_a4_margins_and_compact_table_text -q
```

RED: 2 failed. The failures showed reversed `Table 2`--`Table 5` caption offsets and missing cell no-wrap formatting. Restoring the minimal builder changes made the same command pass.

The final efficiency-layout behavior was then driven separately:

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  thesis/tests/test_build_review_documents.py::test_format_review_docx_sets_a4_margins_and_compact_table_text -q
```

RED: the wide table still had `w:tblLayout w:type="fixed"` instead of `autofit`. GREEN after the wide-table-only auto-fit change:

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  thesis/tests/test_build_review_documents.py::test_format_review_docx_sets_a4_margins_and_compact_table_text \
  thesis/tests/test_build_review_documents.py::test_assemble_review_markdown_preserves_table_order_for_links_in_one_paragraph -q
```

Result: 2 passed.

## Export and artifact verification

The review DOCX files were rebuilt with Pandoc and exported through Microsoft Word's PDF exporter. Final outputs are A4: English DOCX/PDF 1,715,633/1,389,383 bytes and 24 PDF pages; Chinese DOCX/PDF 1,719,039/1,525,457 bytes and 22 PDF pages.

`inspect_docx()` found, in each language, 5 tables, equations 1--20, 10 media files, 8 drawings, a reference heading, no unresolved citations, and no external table links. Exact DOCX text checks confirmed table-caption order 1--5 and figure-caption order 1--8 in both languages. PDF text checks found no split `Strategy`, `Baseline`, or `Pyramid` labels in the efficiency table.

Rendered-page inspection covered English Figure 1, English Table 5, and Chinese Table 5. Figure captions are explicit and ordered; the efficiency table preserves ordinary labels without mid-word wrapping.

## Validators and tests

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest thesis/tests tests/test_experiment_runner.py -q
# 90 passed, 3 warnings

/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest -q
# 185 passed, 11 warnings

/home/pfy22/miniconda3/envs/loop_splat/bin/python thesis/scripts/validate_bibliography.py \
  thesis/v2/references.bib thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md \
  --published-only --require-doi --require-order
# bibliography validation passed

/home/pfy22/miniconda3/envs/loop_splat/bin/python thesis/scripts/validate_manuscripts.py \
  thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md \
  --bib thesis/v2/references.bib --root thesis/v2 --require-citation-order
# manuscript validation passed

git diff --check
git diff --exit-code 81ee80d -- thesis/manuscript-en.md thesis/manuscript-zh.md thesis/references.bib thesis/review
```

Both Git checks exited zero. The v1 files are byte-identical to `81ee80d`.

## Concerns

The test suite reports existing dependency deprecation warnings from `pynvml`, `distutils`, NumPy matrix usage, and FAISS TypedStorage. They do not fail tests and are outside this final-fix scope.
