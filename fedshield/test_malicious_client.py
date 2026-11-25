"""
test_malicious_client.py - run with:  python test_malicious_client.py

Checks malicious_client.py on CPU with a small fake dataset (no MNIST download,
no federated run; about 10 seconds):
  - every attack mode returns a state_dict with the model's keys, shapes, dtypes
  - the caller's state_dict is never modified
  - same seed and call number -> same upload (reproducible runs and --resume)
  - each attack does what its name says (free_ride, zero_weights, nan, scale vs
    sign_flip, layer_only, label_flip)
  - random_weights does not disturb the simulator's global RNG
  - configure() rejects unknown settings, and level3_fedshield's --attack-setting
    overrides are restored to the defaults before the next run
  - the detector accepts every finite attack as well-formed and rejects the NaN one
  - --malicious-id random gives each attack mode its own client, reproducibly
"""
import torch
from torch.utils.data import TensorDataset

import malicious_client as mc
from common import SmallCNN
from fedshield_detector import extract_delta

DEVICE = torch.device("cpu")
DEFAULTS = dict(mc.ATTACK_CONFIG)


def _setup(mode, **settings):
    mc.ATTACK_CONFIG.clear()
    mc.ATTACK_CONFIG.update(DEFAULTS)
    mc.configure(mode=mode, seed=7, **settings)
    mc.reset_call_count()


def _global_state():
    torch.manual_seed(0)
    return SmallCNN().state_dict()


def _dataset(n=64):
    g = torch.Generator().manual_seed(1)
    return TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                         torch.randint(0, 10, (n,), generator=g))


def _delta(state, gs):
    return {k: state[k] - gs[k] for k in gs if gs[k].is_floating_point()}


def _attack(mode, gs=None, data=None, **settings):
    _setup(mode, **settings)
    gs = gs if gs is not None else _global_state()
    return mc.generate_malicious_update(gs, data or _dataset(), DEVICE)


def test_every_mode_returns_valid_shapes():
    gs = _global_state()
    for mode in mc.ATTACK_MODES:
        out = _attack(mode, gs)
        assert set(out) == set(gs), mode
        for k in gs:
            assert out[k].shape == gs[k].shape, (mode, k)
            assert out[k].dtype == gs[k].dtype, (mode, k)
            assert out[k].device.type == "cpu", (mode, k)
        if mode != "nan":
            assert all(torch.isfinite(v).all() for v in out.values()), mode


def test_input_not_modified():
    gs = _global_state()
    before = {k: v.clone() for k, v in gs.items()}
    for mode in mc.ATTACK_MODES:
        _attack(mode, gs)
        assert all(torch.equal(gs[k], before[k]) for k in gs), mode


def test_reproducible_per_call():
    gs, data = _global_state(), _dataset()
    for mode in ("noise", "sign_flip", "label_flip", "random_weights", "grad_ascent"):
        a = _attack(mode, gs, data)
        b = _attack(mode, gs, data)          # same seed, call counter reset
        assert all(torch.equal(a[k], b[k]) for k in gs), mode
    _setup("noise")
    first = mc.generate_malicious_update(gs, data, DEVICE)
    second = mc.generate_malicious_update(gs, data, DEVICE)   # next round
    assert not torch.equal(first["fc2.weight"], second["fc2.weight"]), \
        "each round's attack should use fresh randomness"


def test_resume_call_count():
    gs, data = _global_state(), _dataset()
    _setup("noise")
    mc.generate_malicious_update(gs, data, DEVICE)
    round2 = mc.generate_malicious_update(gs, data, DEVICE)
    _setup("noise")
    mc.reset_call_count(1)                    # as level3 --resume does after 1 attack call
    resumed = mc.generate_malicious_update(gs, data, DEVICE)
    assert torch.equal(round2["fc2.weight"], resumed["fc2.weight"])


def test_attack_semantics():
    gs, data = _global_state(), _dataset()

    free = _attack("free_ride", gs, data)
    assert all(torch.equal(free[k], gs[k]) for k in gs), "free_ride must return the global model"

    zero = _attack("zero_weights", gs, data)
    assert all((zero[k] == 0).all() for k in gs if gs[k].is_floating_point())

    nan = _attack("nan", gs, data)
    assert torch.isnan(nan["fc2.weight"]).all() and torch.isnan(nan["fc2.bias"]).all()
    assert torch.isfinite(nan["conv1.weight"]).all(), "only the target layer is NaN"

    # scale and sign_flip share the same honest training (same seed), so their
    # updates differ only by the factor: flip_delta / scale_delta = -5 / 10.
    scale = _delta(_attack("scale", gs, data), gs)
    flip = _delta(_attack("sign_flip", gs, data), gs)
    for k in scale:
        assert torch.allclose(flip[k], -0.5 * scale[k], atol=1e-6), k
    assert any(scale[k].abs().sum() > 0 for k in scale), "honest training must change the model"

    # layer_only: honest update everywhere, fc2's update x -10.
    honest = _delta(_attack("scale", gs, data, scale_factor=1.0), gs)
    layer = _delta(_attack("layer_only", gs, data), gs)
    for k in honest:
        expected = -10.0 * honest[k] if k.startswith("fc2.") else honest[k]
        assert torch.allclose(layer[k], expected, atol=1e-5), k


def test_label_flip_labels():
    data = _dataset(20)
    flipped = mc.FlippedLabels(data)
    assert len(flipped) == len(data)
    for i in range(len(data)):
        x, y = data[i]
        fx, fy = flipped[i]
        assert torch.equal(x, fx) and fy == 9 - int(y) and fy != int(y)  # 9 is odd: never y
    gs, d = _global_state(), _dataset()
    honest = _attack("scale", gs, d, scale_factor=1.0)
    lf = _attack("label_flip", gs, d)
    assert not torch.equal(honest["fc2.weight"], lf["fc2.weight"]), \
        "training on flipped labels must give a different model"


def test_random_weights_keeps_global_rng():
    gs, data = _global_state(), _dataset()      # built first: _global_state() seeds the RNG
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    _attack("random_weights", gs, data)
    assert torch.equal(torch.rand(3), expected), "random_weights must not consume the global RNG"


def test_configure_rejects_bad_settings():
    for bad in (dict(mode="not_an_attack"), dict(unknown_key=1)):
        try:
            mc.configure(**bad)
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"configure accepted {bad}")
    _setup("label_flip")


def test_level3_attack_settings_reset():
    import level3_fedshield as l3
    s = l3._parse_attack_settings(["flip_factor=-1", "target_layer=fc1"])
    assert s == {"flip_factor": -1.0, "target_layer": "fc1"}
    assert l3._settings_tag(s) == "_flip_factor-1.0_target_layerfc1"
    for bad in (["flip_factor"], ["seed=3"], ["no_such_key=1"]):
        try:
            l3._parse_attack_settings(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    # What level3 does before each run: defaults + this run's overrides.
    base = dict(mode="sign_flip", seed=1, local_epochs=1, lr=0.01, momentum=0.9)
    mc.configure(**{**l3._ATTACK_DEFAULTS, **s, **base})
    assert mc.ATTACK_CONFIG["flip_factor"] == -1.0
    mc.configure(**{**l3._ATTACK_DEFAULTS, **base})      # next run, no overrides
    assert mc.ATTACK_CONFIG["flip_factor"] == DEFAULTS["flip_factor"]
    assert mc.ATTACK_CONFIG["target_layer"] == DEFAULTS["target_layer"]


def test_random_malicious_id_per_attack():
    import level3_fedshield as l3
    for seed in (42, 7, 123):
        picks = [l3._pick_malicious_id("random", seed, 10, a) for a in mc.ATTACK_MODES]
        assert len(set(picks)) == len(picks), "each attack mode must get its own client"
        assert picks == [l3._pick_malicious_id("random", seed, 10, a) for a in mc.ATTACK_MODES]
    assert l3._pick_malicious_id("5", 42, 10, "noise") == 5          # explicit id unchanged
    assert 0 <= l3._pick_malicious_id("random", 42, 10, "custom") < 10  # TA replacement file


def test_detector_validity_check():
    gs = _global_state()
    for mode in mc.ATTACK_MODES:
        delta, why = extract_delta(gs, _attack(mode, gs))
        if mode == "nan":
            assert delta is None and why, "NaN upload must be rejected as invalid"
        else:
            assert delta is not None, (mode, why)


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
                print("PASS", name)
    finally:
        mc.ATTACK_CONFIG.clear()
        mc.ATTACK_CONFIG.update(DEFAULTS)
