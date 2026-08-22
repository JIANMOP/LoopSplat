# LoopSplat Publication Experiment Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the IMU, Gaussian Pyramid, GI-KF, evaluation, and experiment-runner paths so every reported ablation is physically meaningful, isolated, reproducible, and GPU-verified.

**Architecture:** Keep the LoopSplat pipeline intact and add small explicit boundaries: datasets own timestamp normalization and IMU interval extraction; a focused IMU module owns differentiable SO(3) and preintegration; Tracker owns one prediction/one commit per frame; Mapper receives Pyramid configuration explicitly; evaluation and experiment metadata use pure helper functions that can be tested without launching SLAM. Existing legacy outputs remain read-only and are never treated as formal results.

**Tech Stack:** Python 3.10, PyTorch 2.1.2/CUDA 12.1, NumPy, OpenCV, PyYAML, pytest, existing LoopSplat CUDA rasterizer.

**Spec:** `docs/superpowers/specs/2026-08-22-publication-experiment-repair-design.md`

## Global Constraints

- Do not modify files under `data/` or existing files under `output/`.
- Keep official LoopSplat mechanisms unchanged unless the audit identified a correctness or reproducibility defect.
- Every production behavior change starts with a failing test and ends with the focused test plus the full test suite passing.
- CUDA tensor, SO(3), preintegration, Mapper, and Tracker tests fail when CUDA is unavailable; they are not silently skipped.
- Formal results require clean run directories, a frozen git commit, identical fixed evaluation frame IDs, identical GSR iterations, and at least three seeds for Baseline and the final method.
- FM and Azure without verified ground truth must not emit ATE/RPE.
- Existing flat C-group results are legacy diagnostics only.

---

### Task 1: Test Harness and Effective Configuration Contracts

**Files:**
- Modify: `requirements.txt`
- Modify: `environment.yml`
- Create: `tests/conftest.py`
- Create: `tests/test_effective_config.py`
- Modify: `src/entities/gaussian_slam.py:89-93`
- Modify: `src/entities/mapper.py:25-56`

**Interfaces:**
- Produces: `Mapper(config: dict, pyramid_config: dict, dataset, logger)`.
- Produces: `Mapper.effective_pyramid_config() -> dict`.
- Later tasks consume the explicit Mapper configuration and CUDA test fixtures.

- [ ] **Step 1: Add pytest to both dependency manifests and add a CUDA fixture**

```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture(scope="session")
def cuda_device():
    assert torch.cuda.is_available(), "publication tests require CUDA"
    return torch.device("cuda")
```

- [ ] **Step 2: Write failing tests for top-level Pyramid propagation and validation**

```python
def test_mapper_reports_requested_pyramid_configuration(mapper_factory):
    mapper = mapper_factory({"enabled": True, "num_sub_levels": 2, "uses_per_level": 8})
    assert mapper.effective_pyramid_config() == {
        "enabled": True, "num_sub_levels": 2, "uses_per_level": 8,
    }

@pytest.mark.parametrize("config", [
    {"enabled": 1, "num_sub_levels": 2, "uses_per_level": 8},
    {"enabled": True, "num_sub_levels": 0, "uses_per_level": 8},
    {"enabled": True, "num_sub_levels": 2, "uses_per_level": 0},
])
def test_mapper_rejects_invalid_pyramid_configuration(mapper_factory, config):
    with pytest.raises((TypeError, ValueError)):
        mapper_factory(config)
```

- [ ] **Step 3: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_effective_config.py -v`

Expected: FAIL because Mapper does not accept `pyramid_config` and has no effective-config method.

- [ ] **Step 4: Implement explicit configuration propagation and minimal validation**

```python
class Mapper:
    def __init__(self, config, pyramid_config, dataset, logger):
        if not isinstance(pyramid_config.get("enabled", False), bool):
            raise TypeError("gaussian_pyramid.enabled must be bool")
        num_levels = int(pyramid_config.get("num_sub_levels", 2))
        uses = int(pyramid_config.get("uses_per_level", 8))
        if num_levels < 1 or uses < 1:
            raise ValueError("pyramid levels and uses must be positive")
```

Pass `config.get("gaussian_pyramid", {})` explicitly from `GaussianSLAM`.

- [ ] **Step 5: Run focused tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_effective_config.py -v`

Commit: `fix: propagate and validate pyramid configuration`

---

### Task 2: Dataset Time and Ground-Truth Contract

**Files:**
- Create: `src/entities/imu_types.py`
- Create: `tests/test_dataset_contracts.py`
- Modify: `src/entities/datasets.py:13-45,76-143`
- Modify: `src/entities/datasets_fm.py:27-146`
- Modify: `src/entities/datasets_azure.py:81-173`
- Create: `src/utils/rgbd_registration.py`
- Create: `tests/test_rgbd_registration.py`
- Modify: `configs/FMDataset/fm_base.yaml`
- Modify: `configs/AzureKinect/azure_kinect.yaml`

**Interfaces:**
- Produces: immutable `IMUInterval` with `timestamps_s`, `accelerations`, `angular_velocities`, `dt_s`, `valid`, and `reason`.
- Produces: `BaseDataset.has_ground_truth: bool` and `get_imu_measurements(start_frame_id, end_frame_id) -> IMUInterval` on IMU datasets.
- Produces: all exposed dataset timestamps in seconds.
- Produces: `register_depth_to_color(depth_m, K_depth, K_color, T_color_depth, output_shape) -> np.ndarray` using nearest-depth z-buffering.
- Produces: explicit `T_cam_imu`, `accel_bias`, `gyro_bias`, noise values, gravity magnitude, and time offset in the effective FM configuration.

- [ ] **Step 1: Write failing FM tests using hand-checked timestamp fixtures**

```python
def test_fm_converts_microsecond_timestamps_to_seconds(fm_dataset):
    assert fm_dataset.timestamps[1] - fm_dataset.timestamps[0] == pytest.approx(0.033275, abs=1e-6)

def test_fm_interval_has_interpolated_boundaries_and_strict_time(fm_dataset):
    interval = fm_dataset.get_imu_measurements(0, 1)
    assert interval.valid
    assert interval.timestamps_s[0] == pytest.approx(fm_dataset.timestamps[0])
    assert interval.timestamps_s[-1] == pytest.approx(fm_dataset.timestamps[1])
    assert np.all(np.diff(interval.timestamps_s) > 0)
    assert len(interval.timestamps_s) >= 2

def test_fm_declares_no_ground_truth(fm_dataset):
    assert fm_dataset.has_ground_truth is False
```

The test fixture supplies the first two real FM timestamps as literals so the expected `0.033275` value is not computed by production parsing helpers.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_dataset_contracts.py -v`

Expected: FAIL because timestamps remain microseconds and the interval/GT contract does not exist.

- [ ] **Step 3: Implement normalization and boundary interpolation**

Normalize FM timestamps at load time with `timestamp_us * 1e-6`. For Azure, require `data.timestamp_unit` to be one of `s`, `ms`, `us`, or `ns`; reject unknown units. Build the interval from all samples in `[t0, t1]`, inserting linearly interpolated measurements at both boundaries.

- [ ] **Step 4: Preserve TUM timestamps and declare GT capability**

Store associated TUM RGB timestamps on the dataset and set `has_ground_truth=True`; set FM/Azure to false unless Azure loads and validates a real trajectory.

- [ ] **Step 5: Write a failing synthetic RGB-D registration test**

```python
def test_identity_depth_to_color_registration_preserves_depth():
    depth = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    registered = register_depth_to_color(
        depth, np.eye(3), np.eye(3), np.eye(4), output_shape=(2, 2))
    np.testing.assert_allclose(registered, depth)

def test_depth_registration_keeps_nearest_surface_on_collision():
    registered = register_projected_samples(
        uv=np.array([[0, 0], [0, 0]]), z=np.array([2.0, 1.0]), output_shape=(1, 1))
    assert registered[0, 0] == pytest.approx(1.0)
```

- [ ] **Step 6: Run registration tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_rgbd_registration.py -v`

Expected: FAIL because depth-to-color registration does not exist.

- [ ] **Step 7: Implement calibrated FM depth registration and disable uncalibrated Azure formal runs**

Parse the documented FM color/depth intrinsics and `T_color_depth` into YAML, register raw or filtered depth into the color camera before returning a frame, and record that preprocessing in the manifest. Remove the misleading Azure K4A placeholder from the formal ablation matrix: Azure is eligible again only after a real calibrated transform passes the same synthetic and dataset-shape contracts.

- [ ] **Step 8: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_dataset_contracts.py tests/test_rgbd_registration.py -v`

Commit: `fix: normalize dataset time and ground truth contracts`

---

### Task 3: Differentiable IMU Preintegration

**Files:**
- Create: `src/entities/imu_preintegration.py`
- Create: `tests/test_imu_preintegration.py`
- Modify: `configs/FMDataset/fm_base.yaml`
- Modify: `configs/AzureKinect/azure_kinect.yaml`

**Interfaces:**
- Produces: `so3_exp(rotation_vector: Tensor) -> Tensor`.
- Produces: `so3_log(rotation_matrix: Tensor) -> Tensor`.
- Produces: `preintegrate_imu(interval, bias_accel, bias_gyro, gravity_cam) -> IMUPrediction`.
- Produces: `estimate_gravity(initial_interval, accel_noise_std, stationary_threshold) -> GravityEstimate`.
- `IMUPrediction` contains `delta_R`, `delta_v`, `delta_p`, `total_dt`, `sample_count`, `translation_valid`, and `reason`.

- [ ] **Step 1: Write failing CUDA tests for analytic rotations and gradients**

```python
def test_constant_z_angular_velocity_integrates_to_expected_rotation(cuda_device):
    prediction = preintegrate_fixture(omega=(0.0, 0.0, 1.0), duration=0.2)
    assert so3_log(prediction.delta_R)[2].item() == pytest.approx(0.2, abs=1e-4)

def test_so3_residual_has_finite_nonzero_rotation_gradient(cuda_device):
    rotvec = torch.tensor([0.04, -0.02, 0.01], device=cuda_device, requires_grad=True)
    residual = so3_log(target_R.T @ so3_exp(rotvec))
    residual.square().sum().backward()
    assert torch.isfinite(rotvec.grad).all()
    assert torch.linalg.vector_norm(rotvec.grad) > 0
```

- [ ] **Step 2: Write failing static-IMU translation test**

Use literal acceleration equal to the configured gravity vector and assert `||delta_p|| < 1e-5 m` after compensation.

Add one non-stationary fixture whose acceleration variance exceeds the literal threshold and assert `translation_valid=False` while rotation remains valid.

- [ ] **Step 3: Run tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_imu_preintegration.py -v`

Expected: FAIL because the preintegration module does not exist.

- [ ] **Step 4: Implement pure-PyTorch SO(3), midpoint integration, and finite-value validation**

Use Taylor-safe coefficients around zero, no NumPy/SciPy/detach in the differentiable path, and fixed configured biases. Estimate gravity from the initial interval only when acceleration variance and gyro magnitude satisfy the configured stationary thresholds; otherwise mark translation invalid while retaining the rotation prediction. Transform IMU rotation and specific force with the configured `T_cam_imu` rotation before forming camera-frame residuals.

- [ ] **Step 5: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_imu_preintegration.py -v`

Commit: `feat: add differentiable imu preintegration`

---

### Task 4: Tracker One-Prediction/One-Commit IMU Lifecycle

**Files:**
- Create: `tests/test_tracker_imu_state.py`
- Modify: `src/entities/tracker.py:30-317`
- Modify: `src/entities/gaussian_slam.py:367-385`
- Modify: `src/entities/logger.py`

**Interfaces:**
- Produces: `Tracker.prepare_imu_prediction(frame_id) -> IMUPrediction`.
- Produces: `Tracker.compute_imu_loss(relative_pose, prediction) -> Tensor` with no state mutation.
- Produces: `Tracker.commit_imu_state(frame_id, final_c2w, prediction) -> None` called exactly once after selecting the best pose.

- [ ] **Step 1: Write failing state-lifecycle tests**

```python
def test_repeated_imu_loss_does_not_mutate_tracker_state(tracker, prediction, relative_pose):
    before = tracker.imu_state.copy()
    tracker.compute_imu_loss(relative_pose, prediction)
    tracker.compute_imu_loss(relative_pose, prediction)
    assert tracker.imu_state == before

def test_commit_advances_state_once(tracker, prediction, final_c2w):
    tracker.commit_imu_state(3, final_c2w, prediction)
    once = tracker.imu_state.copy()
    with pytest.raises(RuntimeError):
        tracker.commit_imu_state(3, final_c2w, prediction)
    assert tracker.imu_state == once
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_tracker_imu_state.py -v`

Expected: FAIL because loss mutates velocity and no commit API exists.

- [ ] **Step 3: Replace the old nearest-sample loss path**

Prepare one prediction before the optimization loop, compute Huber-scaled rotation/translation residuals from that immutable prediction in every iteration, then commit only after `best_w2c` is finalized. Remove SciPy rotation conversion and `prev_timestamp` mutation from loss evaluation.

- [ ] **Step 4: Track frame 1 when GT is unavailable**

Use dataset pose directly only for frame 0, or when `tracking.gt_camera=true` and `dataset.has_ground_truth=true`. Call Tracker beginning at frame 1 for FM/Azure.

- [ ] **Step 5: Run focused and regression tests, then commit**

Run: `conda run -n loop_splat python -m pytest tests/test_tracker_imu_state.py tests/test_dataset_contracts.py tests/test_imu_preintegration.py -v`

Commit: `fix: make imu tracking state frame scoped`

---

### Task 5: Validity-Aware Gaussian Pyramid Scheduling

**Files:**
- Create: `tests/test_gaussian_pyramid.py`
- Modify: `src/utils/mapper_utils.py:509-666`
- Modify: `src/entities/mapper.py:123-280`
- Modify: `src/entities/gaussian_slam.py:154-178`

**Interfaces:**
- Produces: `build_depth_pyramid(depth, valid_mask, num_sub_levels) -> list[tuple[Tensor, Tensor]]`.
- Produces: `Mapper.next_pyramid_level(frame_id) -> int` where the full-resolution level is `num_sub_levels`.
- Clears per-submap Pyramid counters during submap reset.

- [ ] **Step 1: Write failing depth-validity and schedule tests**

```python
def test_zero_depth_does_not_become_positive_after_downsampling(cuda_device):
    depth = torch.tensor([[0.0, 0.0], [0.0, 2.0]], device=cuda_device)
    levels = build_depth_pyramid(depth, depth > 0, 1)
    low_depth, low_valid = levels[0]
    assert low_valid.item() is True
    assert low_depth.item() == pytest.approx(2.0)

def test_pyramid_schedule_uses_each_low_level_exactly_n_times(mapper):
    assert [mapper.next_pyramid_level(7) for _ in range(6)] == [0, 0, 1, 1, 2, 2]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_gaussian_pyramid.py -v`

Expected: FAIL because depth resize mixes zero values and the schedule is embedded in the optimizer loop.

- [ ] **Step 3: Implement weighted depth downsampling and explicit schedule**

Downsample `depth * valid` and `valid.float()` using area pooling, divide only where pooled weight is positive, and retain a boolean validity mask. Consume a schedule counter only on iterations that execute an optimizer step.

- [ ] **Step 4: Add effective-state logging and reset**

Record requested/effective Pyramid state and per-level usage counts; clear counters when starting a new submap.

- [ ] **Step 5: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_gaussian_pyramid.py tests/test_effective_config.py -v`

Commit: `fix: make gaussian pyramid valid and observable`

---

### Task 6: GI-KF Isolation and Observable Selection Reasons

**Files:**
- Create: `tests/test_keyframe_selection.py`
- Modify: `src/entities/gaussian_slam.py:180-302`
- Modify: `src/utils/mapper_utils.py:339-503`
- Modify: `configs/FMDataset/fm_base.yaml`
- Modify: `configs/TUM_RGBD/tum_rgbd.yaml`
- Modify: `configs/AzureKinect/azure_kinect.yaml`

**Interfaces:**
- Produces: `KeyframeDecision(selected: bool, score: float, reason: str, components: dict)`.
- GI-KF uses visual motion by default; optional gyro assistance has a separate `keyframing.use_imu_gyro` switch and therefore a separate ablation label.
- Produces structured counters for score, first/last frame, submap boundary, and max-gap forced selections.

- [ ] **Step 1: Write failing isolation tests**

```python
def test_gi_kf_does_not_read_gyro_when_keyframe_imu_is_disabled(selector):
    dataset = DatasetWhoseImuAccessRaises()
    decision = selector(dataset=dataset, use_imu_gyro=False)
    assert isinstance(decision.selected, bool)

def test_fm_visual_velocity_uses_seconds(selector, fm_dataset):
    decision = selector.evaluate_frames(0, 1, translation_m=0.033275)
    assert decision.components["linear_velocity_mps"] == pytest.approx(1.0, rel=1e-3)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_keyframe_selection.py -v`

Expected: FAIL because GI-KF reads gyro unconditionally and returns only a boolean.

- [ ] **Step 3: Decouple gyro and make all forced decisions explicit**

Use dataset seconds, gate gyro access on `keyframing.use_imu_gyro`, expose max-gap as configuration, and record whether a frame was selected by score, first/last rule, submap boundary, or safety rule.

- [ ] **Step 4: Keep the current frustum proxy but name it accurately**

Rename diagnostics from rendered visibility to `frustum_center_overlap`; do not claim occlusion-aware visibility. Keep comparison to the temporal-last keyframe unless a separately tested spatial-nearest policy is introduced.

- [ ] **Step 5: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_keyframe_selection.py tests/test_dataset_contracts.py -v`

Commit: `fix: isolate and audit gi keyframe selection`

---

### Task 7: Loop-Closure Cache Correctness and Uniform GSR Budget

**Files:**
- Create: `tests/test_loop_closure_cache.py`
- Modify: `src/entities/lc.py:65-192`
- Modify: `src/entities/gaussian_slam.py:343-365`
- Modify: `src/gsr/solver.py:101-183`
- Modify: `configs/FMDataset/fm_base.yaml`
- Modify: `scripts/run_ablation.py`

**Interfaces:**
- Produces: `Loop_closure.invalidate_submap_cache(submap_ids) -> None`.
- Every experiment manifest records one effective `gsr_max_iters` value.

- [ ] **Step 1: Write a failing cache invalidation test**

Create a temporary submap checkpoint with one Gaussian position, populate the cache, overwrite the checkpoint with a transformed position, call the PGO correction path, and assert the next load returns the transformed position.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_loop_closure_cache.py -v`

Expected: FAIL because cached pre-PGO Gaussian parameters are reused.

- [ ] **Step 3: Invalidate corrected cache entries and validate GSR iterations**

Invalidate every submap changed by PGO. Require `gsr_max_iters` to be a positive integer and set one shared value in the ablation runner rather than relying on missing-field defaults.

- [ ] **Step 4: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_loop_closure_cache.py -v`

Commit: `fix: invalidate loop closure cache after pgo`

---

### Task 8: Fair Evaluation Without Fake Ground Truth

**Files:**
- Create: `src/evaluation/protocol.py`
- Create: `tests/test_evaluation_protocol.py`
- Modify: `src/evaluation/evaluator.py:36-370`
- Modify: `src/evaluation/evaluate_merged_map.py:19-206`
- Modify: `src/utils/eval_utils.py`
- Modify: `scripts/aggregate_results.py`
- Modify: `configs/FMDataset/fm_base.yaml`
- Modify: `configs/TUM_RGBD/tum_rgbd.yaml`

**Interfaces:**
- Produces: `build_evaluation_frame_ids(num_frames: int, stride: int) -> list[int]` with sorted unique IDs including first and last.
- Produces: `trajectory_status(dataset) -> "available" | "skipped_no_ground_truth"`.
- Writes `evaluation_frame_ids.json`, `trajectory_status.json`, and `rendering_metrics_observed_view.json`.

- [ ] **Step 1: Write failing pure protocol tests**

```python
def test_evaluation_ids_are_fixed_sorted_and_unique():
    assert build_evaluation_frame_ids(10, 3) == [0, 3, 6, 9]

def test_no_gt_dataset_skips_trajectory_files(tmp_path, evaluator_factory):
    evaluator_factory(has_ground_truth=False, output=tmp_path).run_trajectory_eval()
    assert not (tmp_path / "ate.json").exists()
    assert json.loads((tmp_path / "trajectory_status.json").read_text())["status"] == "skipped_no_ground_truth"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_evaluation_protocol.py -v`

Expected: FAIL because the protocol helper and skip status do not exist.

- [ ] **Step 3: Implement fixed observed-view evaluation**

Generate frame IDs only from sequence length and configured stride, never from submap keyframes. Mask invalid depth pixels. Keep the existing keyframe metrics under a diagnostic filename and exclude them from the formal table.

- [ ] **Step 4: Separate map evaluation from optional global refinement**

Record whether refinement is enabled and its iteration count. Formal keyframe-efficiency comparisons must use the same setting and fixed observed frames; the aggregator rejects mismatches.

- [ ] **Step 5: Add TUM RPE and update aggregation**

For GT datasets write ATE, translational RPE, rotational RPE, alignment mode, and valid-pair counts. Read the new observed-view metrics and legacy global metrics only when explicitly requesting legacy diagnostics.

- [ ] **Step 6: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_evaluation_protocol.py -v`

Commit: `fix: use fixed frames for fair slam evaluation`

---

### Task 9: Non-Overwriting Experiment Runs and Provenance

**Files:**
- Create: `src/utils/experiment_utils.py`
- Create: `tests/test_experiment_runner.py`
- Modify: `scripts/run_ablation.py:216-320`
- Modify: `scripts/aggregate_results.py`
- Modify: `run_slam.py`
- Modify: `run_slam_azure.py`
- Modify: `src/entities/gaussian_slam.py:104-119`

**Interfaces:**
- Produces: `create_run_directory(output_root, experiment_id, seed, now_utc=None, suffix=None) -> Path`.
- Produces: `write_manifest(run_dir, config, argv, effective_features) -> dict`.
- Produces: `write_status(run_dir, state, **fields) -> None` where state is `running`, `succeeded`, or `failed`.
- Produces: `discover_completed_runs(output_root) -> list[RunRecord]`.

- [ ] **Step 1: Write failing filesystem behavior tests**

```python
def test_two_runs_never_share_a_directory(tmp_path):
    first = create_run_directory(tmp_path, "C1_0", 0, fixed_time, "aaaa")
    second = create_run_directory(tmp_path, "C1_0", 0, fixed_time, "bbbb")
    assert first != second

def test_only_succeeded_complete_run_is_resumable(tmp_path):
    run = make_run(tmp_path, status="succeeded", include_required_metrics=True)
    assert discover_completed_runs(tmp_path) == [run]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n loop_splat python -m pytest tests/test_experiment_runner.py -v`

Expected: FAIL because run-directory and manifest helpers do not exist.

- [ ] **Step 3: Implement run directories, manifest, status, and logs**

Use `output/ablation/<experiment>/seed_<seed>/<UTC>_<suffix>/`. Manifest includes git commit/dirty state, config SHA-256, command, Python/PyTorch/CUDA/GPU, requested/effective features, GSR iterations, and evaluation frame IDs. Stream subprocess output to both terminal and `run.log`.

- [ ] **Step 4: Fix CLI dead options and output-path validation**

Apply `--gt_camera`, accept seed 0 with `is not None`, and fail clearly when no output path exists instead of reading a missing key. Do not add dotted-key CLI parsing.

- [ ] **Step 5: Run tests and commit**

Run: `conda run -n loop_splat python -m pytest tests/test_experiment_runner.py tests/test_evaluation_protocol.py -v`

Commit: `feat: make ablation runs isolated and reproducible`

---

### Task 10: Local GPU Verification and Experiment Freeze

**Files:**
- Create: `configs/smoke/fm_baseline.yaml`
- Create: `configs/smoke/fm_imu.yaml`
- Create: `configs/smoke/fm_pyramid.yaml`
- Create: `configs/smoke/tum_baseline.yaml`
- Modify: `docs/superpowers/specs/2026-08-22-publication-experiment-repair-design.md`

**Interfaces:**
- Produces: four reproducible smoke configurations using `data.frame_limit`, isolated output roots, seed 0, shared GSR budget, and fixed evaluation stride.
- Produces: one frozen commit hash only after all tests and GPU smokes pass.

- [ ] **Step 1: Run the full CUDA test suite**

Run: `conda run -n loop_splat python -m pytest tests -v`

Expected: all tests pass with CUDA detected and no skips for CUDA behaviors.

- [ ] **Step 2: Run local FM Baseline and IMU GPU smokes**

Run both 5–10 frame configurations and assert status is succeeded, losses are finite, IMU state commits once per eligible frame, and no ATE files are written.

- [ ] **Step 3: Run local FM Pyramid GPU smoke**

Assert manifest effective Pyramid state is true and run logs contain nonzero usage for every configured low-resolution level.

- [ ] **Step 4: Run local TUM Baseline GPU smoke**

Assert trajectory status is available, ATE/RPE outputs exist, fixed observed-view metrics exist, and frame IDs equal the configured protocol.

- [ ] **Step 5: Re-run the full suite and inspect the complete diff**

Run: `conda run -n loop_splat python -m pytest tests -v`

Run: `git diff --check && git status --short && git diff --stat`

- [ ] **Step 6: Commit verification artifacts and freeze the local commit**

Commit: `test: add publication gpu smoke configurations`

Record `git rev-parse HEAD`. Only this exact commit may be pushed and pulled by the server for the next verification phase.
