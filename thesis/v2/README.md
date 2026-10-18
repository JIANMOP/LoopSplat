# OARS-GS v2 manuscript package

This directory holds the journal-neutral v2 submission sources for OARS-GS. The English master is the authoritative source for the completed argument. It is organized around the observation-use question, the derived front-end decisions, their ablations, and their limits. LoopSplat remains the inherited implementation and global-consistency backend.

## Submission sources

| Item | Path | Role |
|---|---|---|
| English master | [manuscript-en.md](manuscript-en.md) | Complete v2 English manuscript. |
| Final bibliography | [references.bib](references.bib) | Formally published, DOI-bearing entries ordered by first citation in the English master. |
| Method source | [sections/method-en.md](sections/method-en.md) | Read-only source for Sections 3.1--3.6; the master carries the submission text. |
| Related-work source | [sections/related-work-en.md](sections/related-work-en.md) | Read-only source for Section 2. |
| Experiment sources | [sections/experiments-en.md](sections/experiments-en.md) and [sections/results-en.md](sections/results-en.md) | Read-only sources for Sections 4.1 and 4.2--4.6. |

Edit the master when revising v2 submission prose. Do not independently revise the section-source files during final assembly, because they are the approved source record for the English method, related-work, and experimental argument.

## Shared read-only assets

The v2 manuscript links to shared result assets rather than duplicating numerical content:

- [v2 figures](figures) contain the editable OARS-GS overview and observation-scheduling diagrams.
- [shared tables](../tables) contain the TUM, Replica, FMDataset, and paired-resource tables derived from the normalized CSV.
- [shared figures](../figures) contain metric plots and qualitative diagnostics.

Do not edit tables, result CSVs, qualitative images, or shared result plots as part of manuscript revision. Regenerate a shared asset from its documented evidence source before changing prose that depends on it.

## Internal evidence records

The following files are review aids, not manuscript attachments:

- [claim-evidence matrix](internal/claim-evidence-matrix.md) traces permitted prose to code, tables, and literature.
- [experiment evidence](internal/experiment-evidence.md) records the fixed-seed calculations, resource-pairing rules, observed-view protocol, and the TUM A3 counterexample.
- [reference audit](internal/reference-audit.md) records formal-publication and DOI verification, final citation order, and retention status.
- [method-code audit](internal/method-code-audit.md) documents the executed method behavior.

Two deferred Task 4 review observations remain visible for the final review: the notation choice for the interpolated-endpoint timestamp differences in Eq. (4), and the operationally redundant maximum-gap check noted below the gate ordering. They are not changed here because the approved method source is preserved for the final review pass.

## Strict validation

Run the English bibliography gate from the repository root with the project environment:

~~~bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python \
  thesis/scripts/validate_bibliography.py \
  thesis/v2/references.bib \
  thesis/v2/manuscript-en.md \
  --published-only --require-doi --require-order

wc -w thesis/v2/manuscript-en.md
git diff --check
~~~

The strict bibliography check requires every cited entry to be formally published, DOI-bearing, used by the master, and in exactly the English first-citation order. The word-count target for the English master is 5,500--7,500 words.

When the v2 Chinese master is added, run the bilingual structural gate before producing review documents:

~~~bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python \
  thesis/scripts/validate_manuscripts.py \
  thesis/v2/manuscript-en.md \
  thesis/v2/manuscript-zh.md \
  --bib thesis/v2/references.bib \
  --root thesis \
  --require-citation-order
~~~

## Review copies

After both v2 masters pass validation and Pandoc is available, build journal-neutral review copies without changing either Markdown master:

~~~bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python \
  thesis/scripts/build_review_documents.py \
  --pandoc "$(command -v pandoc)" \
  --thesis-root thesis/v2 \
  --output-dir thesis/v2/review
~~~

Inspect the generated DOCX/PDF files for equation, table, figure, citation, and page-break issues. Apply substantive corrections to the Markdown masters and rebuild; review copies are not independent sources.

## Submission boundary

The v2 submission attachment set is the selected manuscript master, references.bib, the required shared tables and figures, and venue-specific material added after a target journal is chosen. The following are excluded from v2 submission attachments:

- the v1 manuscript and review package under thesis/;
- the opening report;
- thesis/v2/internal/ and all audit records;
- raw result directories, run manifests, configuration hashes, source-control identifiers, and internal review artifacts.

The excluded files remain available for traceability and final review. They are not evidence to package as part of a journal submission.
