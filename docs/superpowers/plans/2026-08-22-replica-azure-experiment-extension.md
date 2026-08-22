# Replica and Azure Experiment Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the eight Replica scenes to the formal ablation matrix, correctly register the self-captured Azure depth into the color camera for a non-formal smoke run, and replace the stale ablation guide.

**Architecture:** Keep dataset-specific normalization in loaders/configs, keep the generic formal runner unchanged except for its declarative scene matrix, and teach the aggregator about a new Replica `R` group. Azure remains outside the formal experiment registry and is exercised only through a smoke config.

**Tech Stack:** Python 3.10, PyTorch/CUDA, OpenCV, PyYAML, pytest, existing LoopSplat runners.

**Spec:** `docs/superpowers/specs/2026-08-22-replica-azure-experiment-extension-design.md`

## Global Constraints

- Formal experiments use seeds `0 1 2`, one frozen commit, one GPU/CUDA environment and `gsr_max_iters=100`.
- Replica uses 8 scenes and 4 non-IMU strategies; FM uses 3 scenes and 6 strategies; TUM uses 5 scenes and 4 strategies.
- Azure is never included in `EXPERIMENTS` and always runs with `tracking.use_imu=false`.
- Replica reconstruction mesh metrics stay disabled while standard culling assets are absent.
- Every production behavior change follows a red-green test cycle.

---

### Task 1: Freeze Replica and Azure data contracts

**Files:**
- Modify: `tests/test_dataset_contracts.py`
- Modify: `tests/test_rgbd_registration.py`
- Modify: `configs/Replica/*.yaml`
- Modify: `configs/AzureKinect/144_5FPS_720p_IMU.yaml`

**Interfaces:**
- Consumes: the user-provided `data/Replica` and `data/AzureKinect/144_5FPS_720p_IMU` layouts.
- Produces: loadable configs whose paths, timestamp units, dimensions and calibration match those files.

- [ ] Add tests asserting every Replica scene config resolves to 2000 RGB, depth and pose entries under `data/Replica/<scene>`.
- [ ] Run those tests and observe failure on the legacy `data/Replica-SLAM` paths.
- [ ] Change only the eight Replica `input_path` values and add explicit formal-safe evaluation defaults to `replica.yaml`.
- [ ] Add an actual-Azure-config test asserting seconds timestamps and the presence of a rigid depth-to-color transform.
- [ ] Run it and observe failure because the current config says `timestamp_unit: us` and does not expose calibrated registration.
- [ ] Update the Azure scene config with the local path, seconds timestamp unit, calibrated preprocessing mode and depth-to-color matrix.
- [ ] Run the focused dataset tests and commit the green state.

### Task 2: Implement calibrated Azure RGB-D registration

**Files:**
- Modify: `src/entities/datasets_azure.py`
- Modify: `tests/test_rgbd_registration.py`
- Create: `configs/smoke/azure_baseline.yaml`

**Interfaces:**
- Consumes: `color_camera`, `depth_camera`, `T_color_depth`, and `preprocessing_strategy: calibrated_depth_to_color`.
- Produces: RGB and depth arrays in the color camera frame with identical dimensions and correct intrinsics.

- [ ] Add a synthetic Azure loader test with different depth/color sizes and a known rigid transform; assert that the registered nearest surface lands in the expected color pixel.
- [ ] Run it and observe failure because the calibrated preprocessing mode does not exist.
- [ ] Implement one calibrated branch: nearest-neighbor undistortion for depth, color undistortion, then `register_depth_to_color()`; do not change legacy resize modes.
- [ ] Add cache identity metadata derived from preprocessing mode and calibration so legacy resize caches cannot be silently reused as calibrated outputs.
- [ ] Re-run the focused test, the existing RGB-D registration tests and dataset-contract tests.
- [ ] Add a 6-frame Azure Baseline smoke config with small mapping/tracking budgets and IMU disabled.
- [ ] Commit the green state.

### Task 3: Replace Azure formal ablations with Replica

**Files:**
- Modify: `scripts/run_ablation.py`
- Modify: `tests/test_experiment_runner.py`
- Create: `configs/smoke/replica_baseline.yaml`

**Interfaces:**
- Produces: `EXPERIMENTS` containing exactly 20 TUM, 32 Replica and 18 FM configurations, with no Azure configuration.

- [ ] Replace the Azure matrix test with assertions for 70 configurations, 32 Replica experiments and zero Azure experiments.
- [ ] Assert Replica IDs `R1_0`--`R8_3`, correct scene configs, IMU disabled, and three-seed dry-run count of 210.
- [ ] Run the tests and observe failure against the old 42-configuration matrix.
- [ ] Add `SCENES_R`, construct the four-strategy Replica group, remove `SCENES_B`, and update CLI documentation/counts using UTF-8 config IO.
- [ ] Add an 8-frame Replica office0 Baseline smoke config with reconstruction disabled.
- [ ] Run focused runner tests and `python scripts/run_ablation.py --dry-run`.
- [ ] Commit the green state.

### Task 4: Aggregate Replica formal results

**Files:**
- Modify: `scripts/aggregate_results.py`
- Modify: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes: completed `R1_0`--`R8_3` run directories.
- Produces: a Replica section with ATE/RPE, observed-view metrics, efficiency metrics and mean±sample-std over three seeds.

- [ ] Add a report test asserting eight Replica scene headings, four strategies, trajectory columns and absence of Azure sections.
- [ ] Run it and observe failure against the old report layout.
- [ ] Extend labels, scene IDs, strategy routing and report rendering for group `R`; remove Azure from result discovery.
- [ ] Run aggregator tests and dry-run the Markdown/terminal entry points against empty output handling.
- [ ] Commit the green state.

### Task 5: GPU smoke, guide, and publication verification

**Files:**
- Modify: `md/ablation-guide.md`

**Interfaces:**
- Consumes: verified smoke commands and the 210-run formal matrix.
- Produces: the single authoritative Chinese execution guide.

- [ ] Run the full pytest suite with local CUDA and record the exact pass count.
- [ ] Run Replica and Azure smoke configs on the local GPU; require exit code zero and `formal_outputs_complete=True` for both output directories.
- [ ] Push the frozen commit, pull it on the server, and repeat full tests plus both GPU smokes when both datasets are present there.
- [ ] Replace `md/ablation-guide.md` with exact environment, smoke, `--dry-run`, single experiment, group A/R/C, full 210-run, resume, validation and aggregation commands.
- [ ] Explain every output artifact and which tables/claims it supports; state Azure is qualitative only and Replica mesh reconstruction is disabled without culling assets.
- [ ] Run command-level checks (`--help`, dry-runs, compileall, full pytest, `git diff --check`) and inspect the final diff against the spec.
- [ ] Commit, merge to `main`, push, pull on the server and verify identical clean commit hashes.
