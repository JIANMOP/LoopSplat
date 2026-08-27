# GI-KF Tracking-Support Implementation Plan

> **Required subskill:** Use `superpowers:executing-plans` to implement this plan task by task. Use `superpowers:test-driven-development` for each behavior change and `superpowers:verification-before-completion` before claiming success.

**Goal:** Separate GI-KF persistent keyframe retention from lightweight Gaussian updates needed to keep tracking stable, without changing IMU behavior or the persistent Gaussian Pyramid path.

**Architecture:** Keep the existing GI decision as the sole authority for persistent keyframes. A deterministic helper recognizes only valid gap-2 GI rejections (`below_threshold` and `high_motion_reject`) as transient support frames. `Mapper.support_update()` optimizes the current full-resolution RGB-D view against the active Gaussian model for a fixed small budget while explicitly disabling Gaussian growth, pruning, pyramid construction, pyramid counters, persistent keyframe storage, and mapping visualization.

**Tech stack:** Python 3.12, PyTorch/CUDA, NumPy, pytest, YAML/JSONL experiment audit files.

---

## Assumptions and non-goals

- `support_update_iterations` is a positive integer and is `20` in every formal GI-KF strategy.
- Only GI-KF runs can execute support updates. GI-disabled behavior remains byte-for-byte equivalent at the control-flow level.
- Support eligibility uses the existing persistent-frame gap, not a new threshold or counter.
- A support frame never becomes a loop-closure/keyframe/submap reference and never changes GI's cached reference pose.
- `Mapper.optimize_submap()` may gain opt-in switches, but its defaults must preserve the current persistent mapping path.
- No file under Tracker/IMU implementation is edited. Pyramid changes are restricted to an explicit `use_pyramid=False` call on the support path.
- This implementation does not retune GI scoring, motion thresholds, stable gap, maximum gap, IMU, or Pyramid parameters.

## Success criteria

1. Eligibility tests prove that only gap-2 `below_threshold` and `high_motion_reject` decisions receive support.
2. Mapper tests prove support uses one transient current view, the configured iteration budget, no pyramid, no pruning, no growth, and no retained keyframe.
3. Run audit tests prove support IDs are unique, in range, disjoint from persistent IDs, and identical in statistics and JSONL.
4. Existing IMU, Pyramid, GI-KF, and experiment tests pass unchanged.
5. A local CUDA smoke reaches output generation without NaN/OOM; a clean server C2_2 seed-0 formal pilot satisfies the thresholds in the approved design spec.

## Task 1: Lock the support-eligibility contract

**Files:**

- Modify: `tests/test_keyframe_selection.py`
- Modify: `src/entities/gaussian_slam.py`

### Step 1: Write failing eligibility tests

Import `should_run_tracking_support` and add a parameterized test with these exact cases:

```python
@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (KeyframeDecision(False, 0.0, "below_threshold", {"frame_gap": 2}), True),
        (KeyframeDecision(False, 0.0, "high_motion_reject", {"frame_gap": 2}), True),
        (KeyframeDecision(False, 0.0, "below_threshold", {"frame_gap": 1}), False),
        (KeyframeDecision(False, 0.0, "min_interval", {"frame_gap": 2}), False),
        (KeyframeDecision(False, 0.0, "invalid_depth", {"frame_gap": 2}), False),
        (KeyframeDecision(True, 0.7, "score", {"frame_gap": 2}), False),
        (KeyframeDecision(True, 0.0, "stable_gap", {"frame_gap": 3}), False),
    ],
)
def test_tracking_support_eligibility(decision, expected):
    assert should_run_tracking_support(decision) is expected
```

Extend an existing low-motion score-decision test to assert:

```python
assert decision.components["frame_gap"] == 2
```

### Step 2: Verify the tests fail for the intended reasons

Run:

```bash
pytest -q tests/test_keyframe_selection.py -k 'tracking_support or frame_gap'
```

Expected: import/contract failure because the helper is absent and score decisions do not yet expose `frame_gap`.

### Step 3: Implement the minimum decision helper

Add next to the GI decision helpers:

```python
def should_run_tracking_support(decision):
    return (
        not decision.selected
        and decision.reason in {"below_threshold", "high_motion_reject"}
        and decision.components.get("frame_gap") == 2
    )
```

Add `"frame_gap": frame_gap` to the merged components returned after `gi_slam_keyframe_decision()`. Do not change any selection score or threshold.

### Step 4: Verify the focused and full selector tests

Run:

```bash
pytest -q tests/test_keyframe_selection.py
```

Expected: all selector tests pass.

## Task 2: Add an isolated Mapper support-update path

**Files:**

- Create: `tests/test_tracking_support_update.py`
- Modify: `src/entities/mapper.py`

### Step 1: Write failing Mapper isolation tests

Construct a minimal `Mapper` object with `Mapper.__new__`, a fake dataset, pre-existing `keyframes`, and non-empty pyramid state. Monkeypatch the keyframe builder and optimizer to record calls. Test:

```python
result = mapper.support_update(
    frame_id=2,
    estimate_c2w=np.eye(4),
    gaussian_model=model,
    iterations=20,
)

assert optimize_call["frame_ids"] == [2]
assert optimize_call["iterations"] == 20
assert optimize_call["use_pyramid"] is False
assert optimize_call["prune"] is False
assert mapper.keyframes == original_keyframes
assert mapper._pyramid_step_counts == original_step_counts
assert mapper._pyramid_level_usage == original_usage
```

Also make `compute_seeding_mask`, `seed_new_gaussians`, `grow_submap`, and pyramid builders raise if invoked. Add a parameterized invalid-budget test for `0`, `-1`, `1.5`, and `True`.

### Step 2: Run the tests and observe the missing API failure

Run:

```bash
pytest -q tests/test_tracking_support_update.py
```

Expected: failure because `support_update()` and the optimizer controls do not exist.

### Step 3: Extract only shared full-resolution keyframe construction

Add a private method:

```python
def _build_keyframe(self, frame_id, estimate_c2w, exposure_ab=None,
                    include_pyramid=True):
    _, gt_color, gt_depth, _ = self.dataset[frame_id]
    if not np.any(np.isfinite(gt_depth) & (gt_depth > 0)):
        raise ValueError(f"mapping frame {frame_id} has no valid depth pixels")
    keyframe = {
        "color": torchvision.transforms.ToTensor()(gt_color).cuda(),
        "depth": np2torch(gt_depth, device="cuda"),
        "render_settings": get_render_settings(
            self.dataset.width, self.dataset.height,
            self.dataset.intrinsics, np.linalg.inv(estimate_c2w)),
        "exposure_ab": exposure_ab,
    }
    if include_pyramid and self._pyramid_enabled:
        keyframe["pyramid_colors"] = build_image_pyramid(
            keyframe["color"], self._pyramid_num_sub_levels)
        depth_levels = build_depth_pyramid(
            keyframe["depth"],
            torch.isfinite(keyframe["depth"]) & (keyframe["depth"] > 0),
            self._pyramid_num_sub_levels)
        keyframe["pyramid_depths"] = [depth for depth, _ in depth_levels]
        keyframe["pyramid_valid_masks"] = [valid for _, valid in depth_levels]
        keyframe["pyramid_render_settings"] = []
        for level in range(self._pyramid_num_sub_levels):
            width, height = get_pyramid_level_dims(
                self.dataset.width, self.dataset.height, level,
                self._pyramid_num_sub_levels)
            keyframe["pyramid_render_settings"].append(
                get_pyramid_render_settings(
                    keyframe["render_settings"], width, height))
        self._pyramid_step_counts.setdefault(frame_id, 0)
    return gt_color, gt_depth, keyframe
```

Replace only the duplicated construction block in `map()` with this helper. Preserve the existing pyramid construction statements exactly.

### Step 4: Add opt-in optimizer isolation switches

Change the signature to:

```python
def optimize_submap(self, keyframes, gaussian_model, iterations=100,
                    use_pyramid=True, prune=True):
```

Use Pyramid only under `self._pyramid_enabled and use_pyramid`, advance counters only under the same condition, and execute both existing pruning points only when `prune` is true. Defaults preserve persistent mapping behavior.

### Step 5: Implement the transient entry point

```python
def support_update(self, frame_id, estimate_c2w, gaussian_model,
                   iterations, exposure_ab=None):
    if type(iterations) is not int or iterations < 1:
        raise ValueError("support update iterations must be a positive integer")
    _, _, keyframe = self._build_keyframe(
        frame_id, estimate_c2w, exposure_ab, include_pyramid=False)
    return self.optimize_submap(
        [(frame_id, keyframe)], gaussian_model, iterations,
        use_pyramid=False, prune=False)
```

Do not call seeding, growth, visual logging, mapping logging, or append to `self.keyframes`.

### Step 6: Verify Mapper isolation and persistent regressions

Run:

```bash
pytest -q tests/test_tracking_support_update.py tests/test_mapper_invalid_depth.py tests/test_gaussian_pyramid.py
```

Expected: all pass; existing persistent Pyramid usage tests remain unchanged.

## Task 3: Integrate support state without contaminating persistent state

**Files:**

- Modify: `tests/test_keyframe_selection.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `src/entities/gaussian_slam.py`

### Step 1: Write failing initialization/statistics tests

Extend `test_run_statistics_count_unique_keyframes_and_submaps()` with:

```python
support_mapping_frame_ids=[2, 2, 4],
support_update_iterations=20,
support_update_elapsed_seconds=1.25,
```

Assert the GI block contains `[2, 4]`, count `2`, iterations `20`, and elapsed `1.25`, while persistent `mapping_frame_ids` and `keyframe_count` remain unchanged.

Add a decision-record test proving JSONL includes `"support_update": true` while `selected` remains false.

### Step 2: Run focused tests and confirm missing-field failures

Run:

```bash
pytest -q tests/test_experiment_runner.py -k run_statistics tests/test_keyframe_selection.py -k record
```

Expected: failures because the new parameters and JSONL field do not exist.

### Step 3: Validate and initialize support state

In `GaussianSLAM.__init__`:

```python
self._gi_support_iterations = kf_cfg.get("support_update_iterations", 20)
if type(self._gi_support_iterations) is not int or self._gi_support_iterations < 1:
    raise ValueError("keyframing.support_update_iterations must be a positive integer")
self.support_mapping_frame_ids = []
self._gi_support_elapsed_seconds = 0.0
```

The list remains empty when GI is disabled.

### Step 4: Integrate one support call per eligible frame

Change `_record_keyframe_decision` to accept `support_update=False` and serialize the boolean. In `run()`:

```python
run_support_update = False
if self._gi_enabled:
    decision = mapping_keyframe_decision(
        self, frame_id, gaussian_model, estimated_c2w,
        starts_new_submap)
    run_support_update = should_run_tracking_support(decision)
    _record_keyframe_decision(
        self, frame_id, decision, support_update=run_support_update)
    if decision.selected:
        self.mapping_frame_ids.append(frame_id)
```

After the persistent mapping block, add an `elif run_support_update` branch. Time the call, invoke `mapper.support_update()` with the already estimated pose/exposure, append only to `support_mapping_frame_ids`, and accumulate elapsed time. Never update `keyframes_info` or call `register_gi_keyframe()` for support frames.

### Step 5: Extend run statistics without changing persistent counts

Add optional support arguments to `build_run_statistics()` and serialize these under `gi_keyframing`:

```python
"support_mapping_frame_ids": unique_support_ids,
"support_update_count": len(unique_support_ids),
"support_update_iterations": support_update_iterations,
"support_update_elapsed_seconds": support_update_elapsed_seconds,
```

Pass live support state from `run()`. Keep top-level `mapping_frame_ids` and `keyframe_count` persistent-only.

### Step 6: Verify integration tests

Run:

```bash
pytest -q tests/test_keyframe_selection.py tests/test_experiment_runner.py -k 'run_statistics or decision'
```

Expected: all selected tests pass.

## Task 4: Make the formal audit reject inconsistent support metadata

**Files:**

- Modify: `tests/test_experiment_runner.py`
- Modify: `src/utils/experiment_utils.py`

### Step 1: Create a valid GI formal-output fixture

Build on `write_valid_formal_outputs()` by setting both manifest GI flags true, creating one JSONL record per frame, and adding a GI statistics block with persistent IDs `[0]`, support IDs `[1]`, support count `1`, iterations `20`, and finite nonnegative support elapsed time.

### Step 2: Write failing consistency tests

Add a passing valid fixture test, then parameterized mutations for:

- duplicate support ID;
- negative or out-of-range support ID;
- overlap with persistent `mapping_frame_ids`;
- count/list mismatch;
- JSONL/list mismatch;
- non-positive or non-integer iteration budget;
- negative or non-finite elapsed time;
- missing/non-boolean `support_update` in JSONL.

Each mutation must make `formal_outputs_complete()` return false.

### Step 3: Implement strict validation

When GI is effective, parse every JSONL line and require integer in-range `frame_id`, boolean `selected`, and boolean `support_update`. Collect the support IDs. Validate:

```python
support_ids = gi_statistics.get("support_mapping_frame_ids")
if (
    not isinstance(support_ids, list)
    or any(type(frame_id) is not int for frame_id in support_ids)
    or support_ids != sorted(set(support_ids))
    or any(frame_id < 0 or frame_id >= statistics["frame_count"]
           for frame_id in support_ids)
    or set(support_ids) & set(statistics.get("mapping_frame_ids", []))
    or support_ids != sorted(logged_support_ids)
):
    return False
```

Also require the count, positive integer budget, and finite nonnegative elapsed time. Do not require support count to be nonzero at validation time; the pilot acceptance check handles that.

### Step 4: Verify formal-output tests

Run:

```bash
pytest -q tests/test_experiment_runner.py -k 'formal_gi or support'
```

Expected: all new valid/invalid audit cases pass.

## Task 5: Make the formal strategy explicit and document outputs

**Files:**

- Modify: `tests/test_experiment_runner.py`
- Modify: `scripts/run_ablation.py`
- Modify: `configs/smoke/fm_keyframing.yaml`
- Modify: `md/ablation-guide.md`

### Step 1: Write failing strategy assertions

For every formal strategy whose `keyframing.enable_gi_slam` is true, assert:

```python
assert overrides["keyframing"]["support_update_iterations"] == 20
```

### Step 2: Add the explicit budget to GI configs

Add `support_update_iterations: 20` to all five formal GI strategy overrides and the FM GI smoke config. Do not change any other strategy parameter.

### Step 3: Document the dual-layer experiment semantics

In `md/ablation-guide.md`, explain that:

- `keyframe_count` counts persistent keyframes only;
- support updates are gap-2 transient current-view optimizations;
- support frames do not seed/prune/build Pyramid/enter loop closure;
- `run_statistics.yaml` and `keyframe_decisions.jsonl` expose the support audit fields;
- formal GI runs use 20 iterations.

Include commands to inspect the support count and ID disjointness from a run directory.

### Step 4: Verify strategy and documentation-related tests

Run:

```bash
pytest -q tests/test_experiment_runner.py -k 'strategy or formal_gi or support'
```

Expected: all pass.

## Task 6: Full local verification and change-scope audit

**Files:** No new production changes expected.

### Step 1: Run focused regressions

```bash
pytest -q \
  tests/test_tracking_support_update.py \
  tests/test_keyframe_selection.py \
  tests/test_gaussian_pyramid.py \
  tests/test_mapper_invalid_depth.py \
  tests/test_experiment_runner.py
```

Expected: all pass.

### Step 2: Run the complete suite and syntax compilation

```bash
pytest -q
python -m compileall -q src scripts tests
```

Expected: full suite passes and compileall exits 0.

### Step 3: Audit prohibited changes and diff shape

```bash
git diff --check
git diff --name-only
git diff -- src/entities/tracker.py src/utils/imu_utils.py
git status --short
```

Expected: no whitespace errors; no Tracker/IMU diff; `paper/` remains untracked and unstaged.

### Step 4: Run local CUDA smoke

Run the repository's existing FM GI smoke command/config on the local GPU. Verify the SLAM process produces `run_statistics.yaml`, the decision JSONL contains support booleans, support/persistent IDs are disjoint, and Pyramid summary counters exclude support frames. If the repository's formal wrapper rejects the run solely because the user-owned untracked `paper/` makes `git_dirty=true`, record that separately from pipeline correctness and do not delete or stage `paper/`.

## Task 7: Commit, synchronize, and run one server pilot

### Step 1: Commit only scoped files

Review the diff, stage the files listed in this plan (never `paper/`), and commit with:

```bash
git commit -m "fix: decouple GI tracking support from keyframes"
```

### Step 2: Push and pull on the server

Push the current branch. On the server:

```bash
cd /root/autodl-tmp/LoopSplat
source /etc/network_turbo
git pull
```

Verify the pulled commit hash equals local HEAD and the server worktree is clean before the formal run.

### Step 3: Run server tests and launch only C2_2 seed 0

In the existing tmux session, run the focused tests first. Then launch the repository's single-experiment ablation command for `C2_2 seed 0`; do not run other scenes or seeds concurrently.

### Step 4: Evaluate against the approved acceptance thresholds

After completion, verify formal output success and extract:

- persistent keyframe rate: 33%–50%;
- support update count: greater than zero;
- peak allocated GPU memory: at most 4.5 GiB;
- SLAM elapsed time: at most 2.85 h;
- PSNR at least 13.57, SSIM at least 0.526, LPIPS at most 0.775, Depth-L1 at most 1.229;
- adjacent translation P95 at most 0.089 m;
- adjacent translation jumps above 1 m: at most one.

If trajectory stability fails, stop and report the architecture as unsuccessful under the approved rule; do not tune gaps or thresholds.
