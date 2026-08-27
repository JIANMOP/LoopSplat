# GI-KF Tracking-Support Update Design

## 1. Context and failure evidence

LoopSplat currently uses one list, `mapping_frame_ids`, for two separate
responsibilities:

1. deciding when the active Gaussian submap must be updated so that the next
   frame can still be tracked; and
2. deciding which frames are retained as long-lived keyframes for submap
   optimization, checkpointing, loop closure, and experiment statistics.

GI-KF reduces `mapping_frame_ids`, so it reduces both persistent keyframes and
the map updates needed by the photometric tracker. On C2 seed 0, the latest
coverage-rescue experiment retained the intended efficiency and rendering
quality but still produced 13 adjacent-pose jumps above 1 m. All 19 jumps above
0.5 m triggered submap boundaries, only three coincided with odometry fallback,
and no loop closure was detected. This shows that further motion-gap tuning is
not an adequate fix.

The design must preserve the already validated IMU and Gaussian Pyramid
strategies. It must not change tracker losses, IMU state, pyramid level
selection, pyramid counters, or persistent-keyframe behavior when GI-KF is
disabled.

## 2. Considered approaches

### A. Continue tuning motion thresholds or keyframe gaps

This is the smallest code change, but several experiments have already exposed
the same trade-off. Immediate high-motion selection retained 73.3% of frames
and was stable but saved little time or memory. Hard rejection and gap-3 rescue
retained 37.9%–41.7% but destabilized tracking. This approach is rejected.

### B. Run normal mapping for support frames and then remove them

This would likely improve tracking, but it would still seed Gaussians, prune the
map, execute the full mapping budget, and advance Pyramid state before dropping
the Python keyframe object. It would blur the distinction between persistent
and support frames and would compromise the efficiency ablation. This approach
is rejected.

### C. Add a transient lightweight tracking-support update

This is the selected approach. A skipped GI candidate at persistent gap 2 gets
a small current-view-only Gaussian optimization. It does not grow or prune
Gaussians, is not retained in any keyframe window, and bypasses Gaussian
Pyramid. The following frame can still be selected as a persistent keyframe by
the unchanged GI policy at gap 3.

## 3. Decision model

The GI keyframe decision remains authoritative for persistent storage:

- first frame, last frame, and submap boundaries are persistent;
- persistent gap 1 is skipped by `min_interval`;
- at persistent gap 2, a frame selected by GI score is persistent;
- at persistent gap 2, `below_threshold` or `high_motion_reject` remains
  non-persistent but receives one tracking-support update;
- at persistent gap 3, `stable_gap` or `high_motion_coverage_rescue` is
  persistent;
- invalid-depth frames never receive persistent or support mapping.

Support eligibility is therefore deterministic and does not introduce a new
motion threshold. It depends only on the existing decision, valid depth, and
the gap already recorded by GI-KF.

The formal support budget is `support_update_iterations: 20`. It is explicit in
every formal GI strategy and validated as a positive integer. This is one fifth
of the normal 100-iteration mapping budget and is applied only to the transient
current view.

## 4. Mapper isolation

`Mapper` will expose a dedicated support-update entry point. Persistent mapping
continues to use `Mapper.map()` unchanged from the caller's perspective.

The support update will:

1. load the current RGB-D frame and construct full-resolution render settings;
2. optimize the active Gaussian model against only that current view for 20
   iterations, using the existing color, depth, and isotropic losses;
3. skip Gaussian seeding and `grow_submap()`;
4. skip opacity pruning;
5. skip Pyramid construction, level selection, counter advancement, and
   lifetime-usage accounting;
6. not append to `Mapper.keyframes`;
7. return scalar timing/loss diagnostics only, allowing all frame tensors to be
   released after the call.

Shared keyframe construction and loss code may be extracted only as required to
avoid two divergent implementations. Existing defaults must preserve the
current persistent mapping behavior exactly.

## 5. SLAM state and output contract

`GaussianSLAM` will maintain `support_mapping_frame_ids` separately from
`mapping_frame_ids`.

Support frames must not be added to:

- `mapping_frame_ids` or reported persistent `keyframe_count`;
- `keyframes_info`;
- `Mapper.keyframes`;
- submap checkpoint `submap_keyframes`;
- GI reference pose cache;
- loop-closure inputs.

`keyframe_decisions.jsonl` will keep the original GI decision and add a
`support_update` boolean. `run_statistics.yaml` will add support frame IDs,
count, configured iteration budget, and elapsed time. Formal validation will
require these fields for GI runs and verify that support IDs are unique,
in-range, disjoint from persistent keyframes, and consistent with the JSONL
records.

When GI-KF is disabled, support state is empty and no new Mapper path is called.

## 6. IMU and Pyramid non-interference

No change is permitted in `Tracker`, IMU preintegration/loss/state, or their
formal configuration. Support updates consume the already estimated camera
pose and cannot read IMU data.

No support update may call image/depth pyramid builders, pyramid render
settings, `_current_pyramid_level()`, `_advance_pyramid_level()`, or modify any
pyramid state dictionary. Persistent mapping with Pyramid enabled must produce
the same per-keyframe level progression as before this change.

Tests will explicitly fail if a support update appends a keyframe, grows or
prunes Gaussians, or changes Pyramid counters.

## 7. Testing strategy

Implementation follows test-driven development:

1. unit-test support eligibility for score rejection, motion rejection,
   persistent selection, minimum interval, and invalid depth;
2. unit-test that support updates use the configured iteration count but do not
   seed, grow, prune, retain keyframes, or advance Pyramid state;
3. unit-test the new statistics and formal-output consistency checks;
4. retain all existing IMU, Pyramid, GI decision, experiment-runner, and formal
   output tests;
5. run the complete local test suite and an FM GPU smoke test;
6. audit the Git diff to ensure no IMU or Tracker files changed and all Pyramid
   changes are limited to an explicit bypass used only by support updates.

## 8. Server validation and acceptance

After local verification, commit and push, pull the clean commit on the server,
and run only `C2_2 seed 0` in the existing tmux session.

The pilot passes only if all conditions hold:

- formal output validation succeeds with no OOM or NaN;
- persistent keyframe rate remains between 33% and 50%;
- at least one support update is audited;
- peak allocated GPU memory is at most 4.5 GiB;
- SLAM elapsed time is at most 2.85 hours;
- PSNR is at least 13.57, SSIM at least 0.526, LPIPS at most 0.775, and
  Depth-L1 at most 1.229;
- adjacent translation P95 is at most 0.089 m;
- at most one adjacent translation jump exceeds 1 m.

If trajectory stability still fails, no further interval or threshold tweak is
allowed. The GI-KF adaptation will be treated as incompatible with the current
LoopSplat tracking architecture until a separately approved redesign is made.

