#!/usr/bin/env python3
"""Run unchanged SLAM with detached tracking-coverage observations.

Coverage is measured for the initial and final loss-evaluation renders, not
necessarily the selected best pose. Diagnostic overhead is included in runtime.
No loss, optimizer, keyframe rule, IMU state, or pyramid state is modified.
"""

import json
from pathlib import Path
import runpy
import sys

import torch


def coverage_metrics(alpha, depth):
    alpha = alpha.detach().to(dtype=torch.float64)
    depth = depth.detach()
    return {
        'alpha_mean': alpha.mean().item(),
        'alpha_zero_fraction': (alpha == 0).double().mean().item(),
        'alpha_above_0_01_fraction': (alpha > 0.01).double().mean().item(),
        'alpha_above_0_5_fraction': (alpha > 0.5).double().mean().item(),
        'alpha_above_0_98_fraction': (alpha > 0.98).double().mean().item(),
        'positive_depth_fraction': (depth > 0).double().mean().item(),
    }


def install_tracking_observer(module):
    original_render = module.render_gaussian_model
    original_track = module.Tracker.track
    active = None

    def observed_render(*args, **kwargs):
        result = original_render(*args, **kwargs)
        if active is not None:
            snapshot = (result['alpha'].detach(), result['depth'].detach())
            if active['calls'] == 0:
                active['initial'] = snapshot
            active['final'] = snapshot
            active['calls'] += 1
        return result

    def observed_track(self, frame_id, *args, **kwargs):
        nonlocal active
        active = {'calls': 0}
        completed = False
        try:
            result = original_track(self, frame_id, *args, **kwargs)
            completed = True
            return result
        finally:
            state, active = active, None
            row = {
                'frame_id': int(frame_id),
                'completed': completed,
                'render_calls': state['calls'],
                'coverage_pose': 'initial_and_final_loss_evaluation_not_best_pose',
            }
            for label in ['initial', 'final']:
                if label in state:
                    row[label] = coverage_metrics(*state[label])
            if self.tracking_loss_records:
                record = self.tracking_loss_records[-1]
                if record['frame_id'] == frame_id:
                    row['tracking_record'] = dict(record)
            path = Path(self.logger.output_path) / 'tracking_coverage.jsonl'
            with path.open('a', encoding='utf-8') as output:
                output.write(json.dumps(row) + '\n')

    module.render_gaussian_model = observed_render
    module.Tracker.track = observed_track

    def restore():
        module.render_gaussian_model = original_render
        module.Tracker.track = original_track

    return restore


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.entities import tracker
    restore = install_tracking_observer(tracker)
    try:
        runpy.run_path(str(root / 'run_slam.py'), run_name='__main__')
    finally:
        restore()


if __name__ == '__main__':
    main()
