# Bilingual LoopSplat Journal Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce evidence-backed English and Chinese journal manuscript masters, verified references, reproducible tables and figures, and internal audit records for the LoopSplat enhancement study.

**Architecture:** A normalized evidence layer separates raw experiment outputs and verified literature from manuscript prose. Three specialist workstreams prepare methods/formulas, experiment tables/figures, and literature/related work in parallel after the evidence schema is fixed. The primary writer integrates the English master, derives the Chinese version from the same claim-evidence matrix, applies Humanizer separately to both languages, and runs cross-document validation before delivery.

**Tech Stack:** Markdown, BibTeX, Python 3, JSON/YAML/CSV, pandas, NumPy, Matplotlib, pytest, ripgrep, Git, and read-only SSH access to the experiment output server.

**Spec:** `thesis/2026-09-19-journal-paper-writing-design.md`

## Global Constraints

- Use the current implementation and completed A/TUM, R/Replica, and C/FMDataset outputs as the primary evidence.
- Treat Azure Kinect as qualitative engineering evidence only; do not report ATE/RPE or formal IMU ablation for Azure.
- Report the current runs as fixed-seed results and cross-scene macro averages; never describe them as multi-seed averages or invent standard deviations.
- Do not place Git commits or source fingerprints in the manuscript or submission attachments; preserve them only in `thesis/internal/experiment-audit.md`.
- Call the IMU method a preintegration-derived rotation regularizer or IMU-assisted tracking, not tightly coupled visual-inertial SLAM.
- Call the pyramid method image-pyramid-guided coarse-to-fine mapping, not a hierarchical Gaussian or LOD representation.
- Treat LoopSplat loop closure, GS-GS registration, pose-graph optimization, and submap management as inherited baseline components.
- Use only references verified through DOI, CVF, IEEE, ACM, Springer, arXiv, a journal site, or an author project page.
- Do not claim “first,” “SOTA,” “real-time mobile,” universal robustness, or statistical significance without matching evidence.
- Keep formulas, numbers, citations, figure references, and conclusions consistent between the English and Chinese manuscripts.
- Apply Humanizer without adding, deleting, or changing any scientific fact, number, citation, formula, or scope condition.
- Preserve the user's opening-report PDF unchanged.

## Review Focus

- Mixed metric protocols: fixed observed-view rendering must not be described as novel-view synthesis.
- Dataset-specific keyframe budgets: A/C support updates and Replica budget-matched GI-KF must not be conflated.
- Single fixed seed: tables may macro-average scenes but may not display nonexistent variance.
- Scene-dependent regressions: negative results, especially TUM tracking degradation, must remain visible in tables and discussion.
- IMU scope: FMDataset rotation regularization must not be generalized into full visual-inertial state estimation.

---

### Task 1: Freeze the evidence schema and manuscript file layout

**Files:**
- Create: `thesis/README.md`
- Create: `thesis/internal/claim-evidence-matrix.md`
- Create: `thesis/internal/experiment-audit.md`
- Create: `thesis/internal/reference-audit.md`
- Create: `thesis/tables/.gitkeep`
- Create: `thesis/figures/qualitative/.gitkeep`
- Create: `thesis/scripts/.gitkeep`

**Interfaces:**
- Consumes: approved design in `thesis/2026-09-19-journal-paper-writing-design.md`.
- Produces: fixed file responsibilities and a claim-evidence table used by every later task.

- [ ] **Step 1: Write the evidence field definitions**

Add a table to `claim-evidence-matrix.md` with these columns:

```text
claim_id | manuscript_section | supported_claim | experiment_or_source | exact_location | allowed_wording_en | allowed_wording_zh | prohibited_extension
```

Seed it with claims for the inherited LoopSplat backend, rotation-only IMU loss, GI-KF score, tracking-support update, Replica budget matching, pyramid level schedule, TUM/Replica trajectory metrics, FM rendering metrics, and Azure qualitative use.

- [ ] **Step 2: Record experiment boundaries**

Write `experiment-audit.md` with the A/R/C scene-to-ID mapping, strategy suffix mapping, available metrics, fixed-seed rule, hardware-comparison rule, and the internal-only provenance records collected from manifests. Explicitly state that this file is not a submission attachment.

- [ ] **Step 3: Record the reference audit**

Write `reference-audit.md` with verified identifiers for 3DGS, SplaTAM, GS-SLAM, Gaussian Splatting SLAM, Photo-SLAM, LoopSplat, ORB-SLAM3, MM3DGS SLAM, VINGS-Mono, GI-SLAM, and the Gaussian/Laplacian pyramid papers. Mark GI-SLAM as a preprint.

- [ ] **Step 4: Document the file contract**

Write `thesis/README.md` describing which files are manuscript masters, generated assets, internal records, and future journal-template exports. State that `internal/` is excluded from submission.

- [ ] **Step 5: Verify layout and forbidden claims**

Run:

```bash
find thesis -maxdepth 3 -type f | sort
rg -n "tightly coupled|state-of-the-art|first IMU|首次.*IMU|各向同性高斯地图" thesis/internal thesis/README.md
```

Expected: all planned directories and files exist; any forbidden phrase appears only in an explicit warning or prohibited-wording field.

- [ ] **Step 6: Commit**

```bash
git add thesis/README.md thesis/internal thesis/tables/.gitkeep thesis/figures/qualitative/.gitkeep thesis/scripts/.gitkeep
git commit -m "docs: establish paper evidence structure"
```

### Task 2: Build a reproducible result extractor

**Files:**
- Create: `thesis/scripts/build_paper_assets.py`
- Create: `thesis/tests/test_build_paper_assets.py`
- Create: `thesis/tables/result-source.csv`
- Create: `thesis/tables/dataset-summary.csv`

**Interfaces:**
- Consumes: a results root containing `<experiment_id>/seed_0/<run>/status.json`, `manifest.json`, `rendering_metrics_observed_view.json`, and optional `ate_aligned.json`/`rpe.json`.
- Produces: `load_latest_successful_runs(root: Path) -> list[RunRecord]`, `macro_average(records: list[RunRecord]) -> list[SummaryRecord]`, normalized CSV files, and validation errors for missing/duplicate records.

- [ ] **Step 1: Write fixture-based failing tests**

Create temporary A1/R1/C1 result trees in `test_build_paper_assets.py`. Test that the extractor:

```python
def test_selects_latest_successful_run_per_experiment_and_seed(tmp_path): ...
def test_rejects_nonfinite_metrics(tmp_path): ...
def test_preserves_missing_trajectory_as_null_for_fm(tmp_path): ...
def test_macro_average_groups_by_dataset_and_strategy(tmp_path): ...
def test_does_not_compute_seed_standard_deviation_from_scenes(tmp_path): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest thesis/tests/test_build_paper_assets.py -q
```

Expected: FAIL because `build_paper_assets.py` and its interfaces do not exist.

- [ ] **Step 3: Implement the minimal extractor**

Implement immutable records with fields:

```text
dataset_group, scene_id, strategy_id, experiment_id, seed, run_dir,
frame_count, keyframe_count, submap_count, slam_time_hours,
peak_gpu_gib, ate_cm, rpe_translation, rpe_rotation,
psnr, ssim, lpips, depth_l1, support_update_count,
support_update_seconds, gpu_name, cuda_runtime, internal_source_id
```

Select only successful runs with finite rendering/depth metrics. Keep `internal_source_id` in the internal audit output but omit it from all manuscript-facing CSV columns.

- [ ] **Step 4: Run extractor tests**

Run:

```bash
python -m pytest thesis/tests/test_build_paper_assets.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Extract server results without GPU use**

Run the script against `/root/autodl-fs/output/ablation` on the no-card server, then copy only the generated CSV and compact audit JSON back to `thesis/tables/` and `thesis/internal/`. Do not copy checkpoints, datasets, or rendered image folders.

- [ ] **Step 6: Independently spot-check records**

Compare at least one A, one R, and one C record against the corresponding source JSON/YAML. Confirm that C records have blank trajectory metrics and that all records use seed 0.

- [ ] **Step 7: Commit**

```bash
git add thesis/scripts/build_paper_assets.py thesis/tests/test_build_paper_assets.py thesis/tables/result-source.csv thesis/tables/dataset-summary.csv thesis/internal/experiment-audit.md
git commit -m "feat: extract auditable paper results"
```

### Task 3: Generate tables and quantitative figures

**Files:**
- Modify: `thesis/scripts/build_paper_assets.py`
- Modify: `thesis/tests/test_build_paper_assets.py`
- Create: `thesis/tables/tum-results.md`
- Create: `thesis/tables/replica-results.md`
- Create: `thesis/tables/fm-results.md`
- Create: `thesis/tables/efficiency-results.md`
- Create: `thesis/figures/metric-deltas.pdf`
- Create: `thesis/figures/metric-deltas.png`
- Create: `thesis/figures/quality-efficiency.pdf`
- Create: `thesis/figures/quality-efficiency.png`

**Interfaces:**
- Consumes: `result-source.csv` and `dataset-summary.csv` from Task 2.
- Produces: Markdown tables and publication-resolution vector/raster plots with matching numbers.

- [ ] **Step 1: Write failing table and plot tests**

Add tests that assert:

```python
def test_tables_include_every_scene_and_macro_average(): ...
def test_metric_arrows_match_metric_direction(): ...
def test_relative_changes_use_baseline_as_denominator(): ...
def test_figures_are_generated_in_pdf_and_png(): ...
def test_efficiency_comparisons_do_not_cross_hardware_within_scene(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest thesis/tests/test_build_paper_assets.py -q
```

Expected: the new tests FAIL because generation functions are absent.

- [ ] **Step 3: Implement table generation**

Generate separate tables for TUM, Replica, and FM. Include per-scene rows and a macro-average row. Use `↑` for PSNR/SSIM and `↓` for ATE/RPE/LPIPS/Depth L1/time/memory/keyframes. Do not create a standard-deviation column.

- [ ] **Step 4: Implement plot generation**

Generate:

- paired per-scene metric deltas relative to Baseline;
- a quality-efficiency plot using PSNR versus SLAM time and keyframe count;
- dataset facets so TUM, Replica, and FM protocols are not merged into one scale;
- accessible colors, 300 dpi PNG, embedded-font PDF, and no decorative title inside the plot.

- [ ] **Step 5: Run tests and visual checks**

```bash
python -m pytest thesis/tests/test_build_paper_assets.py -q
python thesis/scripts/build_paper_assets.py --input-csv thesis/tables/result-source.csv --output-root thesis
file thesis/figures/*.pdf thesis/figures/*.png
```

Expected: tests PASS; four plot files are valid; generated Markdown values match the CSV.

- [ ] **Step 6: Commit**

```bash
git add thesis/scripts thesis/tests thesis/tables thesis/figures/metric-deltas.* thesis/figures/quality-efficiency.*
git commit -m "docs: generate experiment tables and figures"
```

### Task 4: Build and verify the bibliography

**Files:**
- Create: `thesis/references.bib`
- Modify: `thesis/internal/reference-audit.md`
- Create: `thesis/scripts/validate_bibliography.py`
- Create: `thesis/tests/test_validate_bibliography.py`

**Interfaces:**
- Consumes: primary-source metadata collected in the reference audit.
- Produces: 25–35 verified BibTeX entries and `validate_bibliography(bib_path: Path, manuscripts: list[Path]) -> list[str]`.

- [ ] **Step 1: Write bibliography validation tests**

Test duplicate keys, missing title/author/year, unmarked preprints, malformed DOI values, citation keys absent from the bibliography, and bibliography entries unused by both manuscripts.

- [ ] **Step 2: Verify tests fail**

```bash
python -m pytest thesis/tests/test_validate_bibliography.py -q
```

Expected: FAIL before the validator exists.

- [ ] **Step 3: Create verified BibTeX entries**

Include the core sources named in the design and enough RGB-D SLAM, loop closure, keyframing, evaluation, and multiscale references to reach 25–35 relevant entries. Use formal venue metadata when available. Keep GI-SLAM and other unreviewed works as `@misc` with arXiv identifiers.

- [ ] **Step 4: Implement and run validation**

```bash
python -m pytest thesis/tests/test_validate_bibliography.py -q
python thesis/scripts/validate_bibliography.py thesis/references.bib
```

Expected: tests PASS; no duplicate or incomplete entries.

- [ ] **Step 5: Commit**

```bash
git add thesis/references.bib thesis/internal/reference-audit.md thesis/scripts/validate_bibliography.py thesis/tests/test_validate_bibliography.py
git commit -m "docs: add verified paper bibliography"
```

### Task 5: Draft the method, formulas, and system diagrams

**Files:**
- Create: `thesis/sections/method-en.md`
- Create: `thesis/sections/method-zh.md`
- Create: `thesis/figures/system-overview.svg`
- Create: `thesis/figures/keyframe-flow.svg`
- Modify: `thesis/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: current implementation, design equations, and verified baseline references.
- Produces: bilingual method sections with identical equation numbering and two editable vector diagrams.

- [ ] **Step 1: Trace every equation to code or a cited source**

Record exact code locations for Gaussian parameters, visual tracking loss, IMU preintegration, rotation residual, Huber loss, keyframe score, support decision, pyramid schedule, and mapping loss in the claim-evidence matrix.

- [ ] **Step 2: Write the English method section**

Use Sections 3.1–3.6 from the design. Include 9–11 equation groups, define every symbol, and distinguish inherited LoopSplat behavior from adapted and newly added behavior.

- [ ] **Step 3: Write the Chinese method section**

Translate by meaning rather than sentence order. Preserve equation numbers, variables, parameter values, citations, and method boundaries.

- [ ] **Step 4: Draw system and keyframe diagrams**

Create SVG diagrams with consistent component names in English. Provide bilingual captions in the section files. Show that support frames do not enter persistent keyframes, pyramid state, submap checkpoints, or loop closure.

- [ ] **Step 5: Verify equations and terminology**

```bash
rg -n "tightly coupled|hierarchical Gaussian|isotropic Gaussian map|紧耦合|层次化高斯地图|各向同性高斯地图" thesis/sections/method-*.md
rg -o "\\tag\{[0-9]+\}" thesis/sections/method-en.md | sort
rg -o "\\tag\{[0-9]+\}" thesis/sections/method-zh.md | sort
```

Expected: forbidden terms absent except in explicit distinctions; equation-tag lists are identical.

- [ ] **Step 6: Commit**

```bash
git add thesis/sections thesis/figures/system-overview.svg thesis/figures/keyframe-flow.svg thesis/internal/claim-evidence-matrix.md
git commit -m "docs: draft bilingual method and diagrams"
```

### Task 6: Draft related work and experimental analysis in parallel

**Files:**
- Create: `thesis/sections/related-work-en.md`
- Create: `thesis/sections/related-work-zh.md`
- Create: `thesis/sections/experiments-en.md`
- Create: `thesis/sections/experiments-zh.md`
- Create: `thesis/sections/results-en.md`
- Create: `thesis/sections/results-zh.md`
- Modify: `thesis/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: verified bibliography, generated tables/figures, experiment audit, and method terminology.
- Produces: bilingual literature synthesis, reproducible setup, and results discussion with every quantitative claim linked to a table row.

- [ ] **Step 1: Write related work by research question**

Organize the section around Gaussian SLAM, global consistency, inertial Gaussian SLAM, keyframing/map updates, and multiscale optimization. Cite MM3DGS and VINGS-Mono to constrain novelty claims. Avoid a paper-by-paper chronology.

- [ ] **Step 2: Write experimental setup**

Describe datasets, scenes, strategy matrices, hardware, fixed seed, metrics, observed-view protocol, and dataset-specific GI-KF budgets. State that FM has no trajectory ground truth and Azure is qualitative.

- [ ] **Step 3: Write results from generated assets**

For every dataset:

- report macro averages and representative per-scene changes;
- retain regressions and explain their observed conditions without inventing causes;
- separate quality, trajectory, time, memory, keyframe, and submap conclusions;
- treat the IMU contribution as mixed if metrics disagree;
- avoid comparing absolute runtime across different GPU models.

- [ ] **Step 4: Write the Chinese counterparts**

Keep all numerical claims, citations, table labels, and scope statements aligned with the English sections.

- [ ] **Step 5: Audit quantitative sentences**

For every sentence containing `%`, `dB`, `cm`, `GiB`, `hour`, or a decimal metric, add its exact CSV/table location to the claim-evidence matrix.

- [ ] **Step 6: Verify no unsupported statistics**

```bash
rg -n "standard deviation|mean ±|statistically significant|p[ -]?value|标准差|均值.*标准差|统计显著" thesis/sections
```

Expected: no claim of unavailable repeated-run statistics.

- [ ] **Step 7: Commit**

```bash
git add thesis/sections thesis/internal/claim-evidence-matrix.md
git commit -m "docs: draft bilingual related work and experiments"
```

### Task 7: Assemble the complete English manuscript

**Files:**
- Create: `thesis/manuscript-en.md`
- Create: `thesis/sections/front-back-en.md`
- Modify: `thesis/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: English method, related-work, experiment, result sections, bibliography, tables, and figures.
- Produces: a coherent 5,500–7,500-word English journal master.

- [ ] **Step 1: Write introduction and contribution boundaries**

Introduce the tracking–mapping–efficiency problem, position LoopSplat, motivate the three strategies, and state the three approved contributions without claiming primacy.

- [ ] **Step 2: Write abstract after the results**

Use 180–230 words. Include only dataset-level findings that can be recomputed from `dataset-summary.csv`. Mention the main tradeoff if one strategy improves efficiency while degrading a metric.

- [ ] **Step 3: Write limitations and conclusion**

Cover fixed-seed uncertainty, rotation-only IMU scope, missing FM trajectory ground truth, observed-view evaluation, and scene-dependent effects. End the conclusion on a concrete supported finding.

- [ ] **Step 4: Assemble sections and assets**

Use one heading hierarchy, continuous equation numbering, stable table/figure labels, and Pandoc-compatible citations such as `[@zhu2025loopsplat]`.

- [ ] **Step 5: Check length and structure**

```bash
wc -w thesis/manuscript-en.md
rg -n '^# ' thesis/manuscript-en.md
rg -n 'Figure [0-9]+|Table [0-9]+|Eq\. \([0-9]+\)' thesis/manuscript-en.md
```

Expected: 5,500–7,500 words; all required sections present; every referenced asset exists.

- [ ] **Step 6: Commit**

```bash
git add thesis/manuscript-en.md thesis/sections/front-back-en.md thesis/internal/claim-evidence-matrix.md
git commit -m "docs: assemble English journal manuscript"
```

### Task 8: Assemble the complete Chinese manuscript

**Files:**
- Create: `thesis/manuscript-zh.md`
- Create: `thesis/sections/front-back-zh.md`

**Interfaces:**
- Consumes: the approved English master and all bilingual section drafts.
- Produces: a 10,000–15,000-character Chinese master with matching evidence and structure.

- [ ] **Step 1: Write the Chinese introduction, abstract, limitations, and conclusion**

Use natural Chinese academic prose. Keep technical terms consistent: 持久关键帧、轻量跟踪支撑、固定观测视角、惯性旋转正则、图像金字塔粗到细建图。

- [ ] **Step 2: Assemble all Chinese sections**

Preserve formula numbering, citations, table/figure labels, numbers, units, and direction of every comparison.

- [ ] **Step 3: Run bilingual structural checks**

```bash
rg -n '^##? ' thesis/manuscript-en.md
rg -n '^##? ' thesis/manuscript-zh.md
rg -o '\[@[^]]+\]' thesis/manuscript-en.md | sort -u
rg -o '\[@[^]]+\]' thesis/manuscript-zh.md | sort -u
```

Expected: corresponding section trees and identical citation-key sets.

- [ ] **Step 4: Commit**

```bash
git add thesis/manuscript-zh.md thesis/sections/front-back-zh.md
git commit -m "docs: assemble Chinese journal manuscript"
```

### Task 9: Apply Humanizer to both manuscript masters

**Files:**
- Modify: `thesis/manuscript-en.md`
- Modify: `thesis/manuscript-zh.md`

**Interfaces:**
- Consumes: complete evidence-checked manuscripts.
- Produces: natural academic prose with unchanged facts, numbers, equations, citations, and conclusions.

- [ ] **Step 1: Snapshot factual tokens before editing**

Extract equations, citations, percentages, decimal numbers, dataset names, strategy names, table labels, and figure labels from both manuscripts to temporary comparison files outside the repository.

- [ ] **Step 2: Run Humanizer file mode on the English manuscript**

Remove staged openings, not-X-but-Y constructions, one-line closers, forced triads, excessive dashes, inflated significance, sales language, decorative bold, and chatbot residue. Preserve neutral technical hedging where uncertainty is real.

- [ ] **Step 3: Run Humanizer file mode on the Chinese manuscript**

Remove equivalent Chinese templates such as “不仅……而且……”, “值得注意的是”, empty “综上所述” repetitions, exaggerated significance, and mechanical translation order. Preserve scientific qualifications and dataset-specific limitations.

- [ ] **Step 4: Compare factual tokens after editing**

Run the same extraction and diff the before/after files. Resolve every changed number, citation, equation tag, unit, table label, or figure label.

- [ ] **Step 5: Scan common AI-writing residue**

```bash
rg -n "—|–|not only|not merely|it is worth noting|delve|pivotal|crucial|landscape|值得注意的是|不仅.*而且|具有重要意义|综上所述" thesis/manuscript-en.md thesis/manuscript-zh.md
```

Expected: remaining matches are justified technical uses or removed.

- [ ] **Step 6: Commit**

```bash
git add thesis/manuscript-en.md thesis/manuscript-zh.md
git commit -m "docs: humanize bilingual journal manuscripts"
```

### Task 10: Final manuscript verification and handoff

**Files:**
- Modify: `thesis/scripts/validate_bibliography.py`
- Create: `thesis/scripts/validate_manuscripts.py`
- Create: `thesis/tests/test_validate_manuscripts.py`
- Modify: `thesis/README.md`

**Interfaces:**
- Consumes: both manuscripts, bibliography, tables, figures, and claim-evidence matrix.
- Produces: a zero-error validation run and a handoff README explaining how to review and convert the masters.

- [ ] **Step 1: Write validation tests**

Test missing assets, mismatched citation sets, mismatched equation tags, unsupported placeholder markers, forbidden claims, unmatched table/figure references, and changed metric numbers between bilingual versions.

- [ ] **Step 2: Run tests to verify they fail before implementation**

```bash
python -m pytest thesis/tests/test_validate_manuscripts.py -q
```

Expected: FAIL because `validate_manuscripts.py` does not exist.

- [ ] **Step 3: Implement the validator**

The command must return nonzero for any missing asset, unresolved citation, `TODO`/`TBD` marker, equation mismatch, or prohibited unsupported claim. It must emit file and line references for each issue.

- [ ] **Step 4: Run all focused and repository tests**

```bash
python -m pytest thesis/tests -q
python thesis/scripts/validate_bibliography.py thesis/references.bib thesis/manuscript-en.md thesis/manuscript-zh.md
python thesis/scripts/validate_manuscripts.py thesis/manuscript-en.md thesis/manuscript-zh.md --bib thesis/references.bib --root thesis
python -m pytest -q
```

Expected: all commands exit 0.

- [ ] **Step 5: Perform a final evidence review**

Sample every dataset and every strategy family. Recompute all abstract percentages from CSV, confirm each reference through its recorded primary URL, and verify that negative results remain disclosed.

- [ ] **Step 6: Update the handoff README**

Document the English and Chinese master paths, generated-asset commands, internal files excluded from submission, and the steps required once a target journal template is selected.

- [ ] **Step 7: Request independent whole-manuscript review**

Use a fresh reviewer agent to check technical accuracy, paper structure, citation integrity, numerical consistency, claim strength, and Humanizer regressions. Address every critical and important finding before completion.

- [ ] **Step 8: Commit**

```bash
git add thesis
git commit -m "docs: finalize bilingual LoopSplat journal paper"
```
