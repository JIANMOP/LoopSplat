# LoopSplat enhanced-strategy journal-paper masters

This directory holds the reviewable source material for the LoopSplat enhanced-strategy journal paper. Markdown is the master format; numerical results must be generated from recorded experiment outputs rather than manually re-entered.

## File contract

| Category | Files | Responsibility |
|---|---|---|
| Manuscript masters | `manuscript-en.md`, `manuscript-zh.md`, `references.bib` | The English and Chinese manuscript masters and their shared, verified bibliography. They will be created in later tasks. |
| Generated assets | `tables/result-source.csv`, `tables/*.md`, `figures/*.pdf`, `figures/qualitative/` | Program-generated result sources, tables, diagrams, and qualitative figures. `tables/`, `figures/qualitative/`, and `scripts/` are kept in Git before their generated contents exist. |
| Internal records | `internal/claim-evidence-matrix.md`, `internal/experiment-audit.md`, `internal/reference-audit.md` | Claim wording, experiment provenance, and reference verification used to keep the masters auditable. `internal/` is excluded from submission. |
| Future journal-template exports | A future journal-specific LaTeX or Word export directory, created only after the target venue is selected | A rendered submission copy derived from the Markdown masters; it is not an independent source of truth. |

The source PDF `thesis/彭飞扬学术学位硕士学位论文开题报告.pdf` is historical context only. It is not a manuscript master and must not be modified by this workflow.

## Submission boundary

Only the manuscript masters, bibliography, and selected generated assets may enter a submission package. The `internal/` directory, source-code hashes, source fingerprints, manifests, and other provenance records remain internal.
