# Loop-Closure Memory Residency Design

## Goal

Remove the loop-closure GPU-memory growth that prevents long TUM and Replica
sequences from completing, while preserving the existing LoopSplat algorithm,
configuration, registration budget, selected views, losses, and evaluation
protocol.

## Scope

This change is limited to the lifetime and placement of loop-closure camera
observations:

- historical RGB observations remain on CPU while the pose graph is built;
- pose-graph analysis cameras do not retain RGB, depth, or gradient masks;
- GSR selects registration views from the unchanged descriptors before camera
  copies are created;
- only the selected source and target cameras are materialized on GPU.

The implementation must not change IMU tracking, Gaussian-pyramid scheduling,
GI-KF selection, loop-candidate detection, GSR iteration counts, registration
losses, or formal evaluation outputs.

## Current Failure

`Loop_closure.construct_pose_graph()` loads every historical keyframe camera.
For datasets without `get_processed_image_paths`, `_make_camera()` immediately
moves each RGB observation and its gradient mask to CUDA. The method then makes
a deep copy of every camera for `cam_dict`. Finally,
`gaussian_registration()` deep-copies the complete camera lists for the source
and target submaps before selecting at most two views from each list.

On TUM `fr2/xyz`, the failed runs contained about 3,000 keyframes. The duplicate
camera observations filled 31.45 GiB before GSR requested its next 20 MiB CUDA
allocation.

## Design

### CPU-resident observations

For datasets whose observations are returned as arrays, `_make_camera()` stores
the RGB observation as a contiguous CPU tensor and keeps depth on CPU. It does
not compute the gradient mask at construction time. Path-backed datasets retain
their existing path-based behavior.

`Camera.load_rgb()` gains the ability to materialize an existing CPU RGB tensor
on CUDA. Integer observations are converted to the same float32 `[0, 1]` range
used by the current implementation. The gradient mask is computed only after
the selected camera reaches CUDA. Existing path-backed loading remains
unchanged.

### Pose-only analysis cameras

`construct_pose_graph()` continues to build `cam_dict`, because PGO analysis
updates those poses. Each entry is still an independent camera copy, but its
RGB, depth, and gradient-mask fields are cleared immediately. This preserves
pose mutation isolation without duplicating observations.

### Select before copying

`gaussian_registration()` computes exactly the current descriptor similarity
matrix and top-k indices before copying cameras. It then deep-copies only the
selected source and target cameras. The Gaussian models remain independent
copies as in the current code, and localization, rendering, loss construction,
iteration count, probability weighting, and transformation estimation remain
unchanged.

At most two source and two target observations are resident on CUDA for a
registration pair. Temporary selected views are released when registration
returns.

## Compatibility and Provenance

The change is an implementation-level memory optimization, but it changes the
formal source fingerprint. Existing successful runs remain preserved under
their original manifests. A short completed scene will be rerun to compare the
old and new outputs before formal A4 experiments begin.

A4's four configurations will be rerun on the same new source and the same GPU.
Replica will only be scheduled after the A4 validation passes.

## Error Handling

- CPU observations must have a supported tensor or NumPy representation.
- `Camera.load_rgb()` must reject unsupported shapes or dtypes rather than
  silently changing pixels.
- Selected camera copies remain isolated so a failed or completed localizer
  cannot mutate the historical pose-graph cameras.
- CUDA cache clearing may be used after references are released, but it is not
  treated as the primary fix.

## Verification

1. Unit-test CPU observation residency and CUDA materialization.
2. Unit-test that view selection preserves the existing top-k UIDs and copies
   only selected cameras.
3. Unit-test that `cam_dict` entries are observation-free and pose-independent.
4. Run the complete local test suite and compile checks.
5. Run a local GPU smoke test.
6. On the server, rerun a short old-success scene and compare loop decisions and
   evaluation metrics with the tagged implementation.
7. Run A4_0, A4_1, A4_2, and A4_3 sequentially in a visible tmux session and
   audit formal outputs, peak GPU memory, and error logs.

## Non-goals

- changing the number or definition of keyframes;
- reducing image resolution;
- changing the number of submaps or loop candidates;
- changing GSR optimization or evaluation settings;
- modifying IMU, Gaussian-pyramid, or GI-KF strategy behavior;
- deleting prior experiment outputs.
