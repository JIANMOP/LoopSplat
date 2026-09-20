# LoopSplat enhanced-strategy journal-paper masters

This directory contains the bilingual journal-paper draft and its auditable experiment assets. Markdown is the source format. The journal title, citation style, page layout, and author metadata remain unset until a venue is selected.

## Review entry points

| Item | Path | Purpose |
|---|---|---|
| English master | [manuscript-en.md](manuscript-en.md) | Complete English paper, approximately 6,400 words. |
| Chinese master | [manuscript-zh.md](manuscript-zh.md) | Complete Chinese counterpart with matching formulas, citations, metrics, tables, and figures. |
| English review copy | [review/manuscript-en-review.docx](review/manuscript-en-review.docx) / [PDF](review/manuscript-en-review.pdf) | A4 copy for reading and annotation; not an independent source. |
| Chinese review copy | [review/manuscript-zh-review.docx](review/manuscript-zh-review.docx) / [PDF](review/manuscript-zh-review.pdf) | A4 copy for reading and annotation; not an independent source. |
| Bibliography | [references.bib](references.bib) | Shared BibTeX library of verified references. |
| Result tables | [tables](tables) | Per-scene TUM, Replica, FMDataset, and efficiency tables plus normalized CSV sources. |
| Figures | [figures](figures) | Editable system diagrams, generated result plots, and saved qualitative diagnostics. |

The paper title is provisional. Tables 2-5 are linked to generated Markdown tables rather than copied into the masters. When a journal template is chosen, place those table bodies in the venue's required positions and apply its caption and width rules.

## Rebuilding result assets

Regenerate the manuscript-facing tables and plots from the normalized result CSV:

~~~bash
python thesis/scripts/build_paper_assets.py \
  --input-csv thesis/tables/result-source.csv \
  --output-root thesis
~~~

To rebuild directly from a complete server result tree, replace the input-CSV option with the result-root path. That route also refreshes the normalized CSV and internal source audit:

~~~bash
python thesis/scripts/build_paper_assets.py \
  /path/to/output/ablation \
  --output-root thesis
~~~

Do not manually edit numerical cells in generated tables. Update the recorded experiment output or normalized CSV, regenerate the assets, and then revise any affected prose through the claim-evidence matrix.

## Validation

Run the focused paper tests and both command-line validators from the repository root:

~~~bash
python -m pytest thesis/tests -q
python thesis/scripts/validate_bibliography.py \
  thesis/references.bib \
  thesis/manuscript-en.md \
  thesis/manuscript-zh.md
python thesis/scripts/validate_manuscripts.py \
  thesis/manuscript-en.md \
  thesis/manuscript-zh.md \
  --bib thesis/references.bib \
  --root thesis
~~~

The manuscript validator checks local assets, citation resolution, bilingual citation sets, formula tags, metric-number sequences, figure/table numbering, placeholders, numbered-section order, and prohibited unsupported claims.

## Building the review copies

The review builder keeps the Markdown masters unchanged. It embeds Tables 2-5, resolves citations, converts display and inline mathematics to editable Word equations, includes the figures, and applies a neutral A4 layout. Run it from the repository root after activating the environment that contains Pandoc and `rsvg-convert`:

~~~bash
conda activate loop_splat
python thesis/scripts/build_review_documents.py \
  --pandoc "$(command -v pandoc)" \
  --thesis-root thesis \
  --output-dir thesis/review
~~~

The command checks that both DOCX files contain all five tables, numbered equations 1-11, at least eight embedded figures, resolved citations, and a reference section. Export the DOCX files from Microsoft Word to refresh the corresponding PDFs, then inspect the wide result tables, equations, figure captions, and page breaks. The review files are convenient reading copies; make substantive edits in `manuscript-en.md` and `manuscript-zh.md`, then rebuild them.

## Converting after journal selection

1. Download the journal's current Word or LaTeX template and author instructions.
2. Copy the English master into that template; use the Chinese master as a meaning and number cross-check.
3. Insert the generated bodies of Tables 2-5, then select the PDF or PNG figure variant required by the venue.
4. Apply the venue's CSL or BibTeX style to references.bib.
5. Add author names, affiliations, declarations, acknowledgements, data/code availability text, and any ethics or funding statements required by the venue.
6. Rerun both validators before submission and visually inspect the rendered PDF for formula, table, font, and figure problems.

## Submission boundary

Submission candidates are the selected manuscript master, references.bib, the four generated result tables, and the required figures. The following remain internal:

- internal/claim-evidence-matrix.md
- internal/experiment-audit.md
- internal/reference-audit.md
- internal/result-source-audit.json
- raw manifests, source identifiers, configuration hashes, and run directories
- the opening report 彭飞扬学术学位硕士学位论文开题报告.pdf

The opening report is historical context only and must not be edited or packaged as a manuscript attachment.
