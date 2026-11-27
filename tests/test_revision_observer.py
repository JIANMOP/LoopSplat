import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def observer_module():
    path = Path(__file__).parents[1] / 'scripts/run_revision_diagnostics.py'
    spec = importlib.util.spec_from_file_location('revision_observer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_detects_empty_render_without_changing_inputs_or_rng():
    module = observer_module()
    alpha = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    depth = torch.tensor([0.0, 2.0, 3.0])
    before = alpha.detach().clone()
    rng = torch.get_rng_state().clone()
    result = module.coverage_metrics(alpha, depth)
    assert result['alpha_zero_fraction'] == 1 / 3
    assert result['alpha_mean'] == 0.5
    assert result['positive_depth_fraction'] == 2 / 3
    assert torch.equal(alpha.detach(), before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert alpha.grad is None


def test_observer_preserves_render_and_tracking_outputs_and_restores(tmp_path):
    module = observer_module()
    rendered = {'alpha': torch.zeros(2, 2), 'depth': torch.ones(2, 2)}
    sentinel = object()

    class Tracker:
        def track(self, frame_id, model, poses):
            assert fake.render_gaussian_model() is rendered
            self.tracking_loss_records.append({'frame_id': frame_id, 'total_loss': 0.0})
            return sentinel

    original_track = Tracker.track
    render = lambda: rendered
    fake = SimpleNamespace(Tracker=Tracker, render_gaussian_model=render)
    tracker = Tracker()
    tracker.logger = SimpleNamespace(output_path=tmp_path)
    tracker.tracking_loss_records = []
    restore = module.install_tracking_observer(fake)
    try:
        assert tracker.track(7, None, None) is sentinel
    finally:
        restore()
    assert Tracker.track is original_track
    assert fake.render_gaussian_model is render
    row = json.loads((tmp_path / 'tracking_coverage.jsonl').read_text())
    assert row['frame_id'] == 7
    assert row['render_calls'] == 1
    assert row['final']['alpha_zero_fraction'] == 1.0
    assert row['tracking_record']['total_loss'] == 0.0
