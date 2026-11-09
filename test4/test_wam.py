"""Unit tests for wam_head.py with tiny fake modules (torch only; no lerobot, no downloads).

    python3 -I test_wam.py

The fake policy mirrors the real data flow that matters here: the flow loss reads the INPUT of
every decoder layer (stand-in for the per-layer K/V the expert consumes); the last layer's own
output and the final norm feed nothing. The head reads layers[-2]'s output = the last layer's
input, so its gradient must reach layers[-2] but never layers[-1].
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import wam_head as W  # noqa: E402

IMG = 7  # fake image_token_id
HID = 32
TDIM = 16


# ------------------------------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------------------------------
class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(HID, HID)

    def forward(self, h):
        return h + torch.tanh(self.mlp(h))


class FakeTextModel(nn.Module):
    """embed -> 3 residual layers -> norm. Records each layer's INPUT (the fake K/V)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(20, HID)
        self.layers = nn.ModuleList([FakeLayer() for _ in range(3)])
        self.norm = nn.LayerNorm(HID)
        self.layer_inputs = []

    def forward(self, input_ids):
        h = self.embed(input_ids) * 5.0  # large-ish, like a pre-norm residual stream
        self.layer_inputs = []
        for layer in self.layers:
            self.layer_inputs.append(h)
            h = layer(h)
        return SimpleNamespace(last_hidden_state=self.norm(h))


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = FakeTextModel()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=HID),
                                      image_token_id=IMG)


class FakePolicy(nn.Module):
    name = "fake"
    config_class = object

    def __init__(self):
        super().__init__()
        self.model = FakeQwen()
        self.expert = nn.Linear(HID, 4)
        self.saved = None

    def forward(self, batch):
        lm = self.model.model.language_model
        lm(batch["input_ids"])
        noise = torch.randn(batch["action"].shape)  # consumes RNG like the real flow forward
        kv = torch.stack([x.mean(1) for x in lm.layer_inputs]).mean(0)
        pred = self.expert(kv)
        flow_loss = ((pred - batch["action"] - noise) ** 2).mean()
        return flow_loss, {"flow_loss": flow_loss.item(), "loss": flow_loss.item()}

    def get_optim_params(self):
        return [{"params": list(self.model.parameters()), "lr": 3e-5},
                {"params": list(self.expert.parameters()), "lr": 5e-5}]

    def _save_pretrained(self, save_directory, state_dict=None):
        self.saved = dict(self.state_dict() if state_dict is None else state_dict)


def make_ids():
    """Row 0: left-pad, text, main image (3 tokens), text, wrist image (2 tokens).
    Row 1: text, main image (4 tokens), text, wrist image (4 tokens), text."""
    return torch.tensor([
        [0, 0, 1, 2, IMG, IMG, IMG, 3, IMG, IMG, 4, 5],
        [1, 2, IMG, IMG, IMG, IMG, 3, IMG, IMG, IMG, IMG, 6],
    ])


def make_table(lengths=(12, 5, 30), dim=TDIM, seed=0):
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    emb = rng.standard_normal((n, dim)).astype(np.float16)
    ep = np.concatenate([np.full(L, e) for e, L in enumerate(lengths)])
    fr = np.concatenate([np.arange(L) for L in lengths])
    return W.TargetTable(emb, ep, fr, horizon=10), emb, ep, fr


def make_batch():
    return {
        "input_ids": make_ids(),
        "attention_mask": (make_ids() != 0).long(),
        "action": torch.randn(2, 4),
        W.EP_KEY: torch.tensor([0, 2]),
        W.FR_KEY: torch.tensor([1, 25]),  # (0,1): 11 < 12 valid; (2,25): 35 >= 30 invalid
    }


def wam_policy(arm, seed=0, table=None, weight=0.1):
    torch.manual_seed(seed)
    policy = FakePolicy()
    table = table if table is not None else make_table()[0]
    cfg = W.WAMConfig(arm=arm, future_weight=weight, head_hidden=24)
    return W.attach_wam(policy, cfg, None if arm == "A" else table), table


# ------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------
def test_head_shapes_and_size():
    head = W.FutureHead(HID, 24, TDIM)
    assert head(torch.randn(5, HID)).shape == (5, TDIM)
    assert sum(p.numel() for p in head.parameters()) == HID * 24 + 24 + 24 * TDIM + TDIM


def test_head_input_norm_is_parameter_free_and_scale_invariant():
    head = W.FutureHead(HID, 24, TDIM)
    nn.init.normal_(head.fc2.weight)
    x = torch.randn(4, HID)
    assert torch.allclose(head(x), head(1000.0 * x + 3.0), atol=1e-4)
    assert set(head.state_dict()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    real = W.FutureHead(2560, 2048, 1024)
    n = sum(p.numel() for p in real.parameters())
    assert 7.0e6 < n < 7.6e6, n  # "about 7M"
    assert torch.count_nonzero(real.fc2.weight) == 0


def test_first_image_run_mask_only_main_camera():
    mask = W.first_image_run_mask(make_ids(), IMG)
    assert mask[0].nonzero().flatten().tolist() == [4, 5, 6]
    assert mask[1].nonzero().flatten().tolist() == [2, 3, 4, 5]


def test_pooling_uses_only_main_image_tokens():
    ids = make_ids()
    hidden = torch.full((2, ids.shape[1], 3), 1000.0)  # poison everything
    main = W.first_image_run_mask(ids, IMG)
    vals = torch.arange(ids.numel(), dtype=torch.float32).reshape(ids.shape)
    hidden[main] = vals[main][:, None].expand(-1, 3)
    pooled, counts = W.pool_main_camera(hidden, ids, IMG)
    assert counts.tolist() == [3, 4]
    assert torch.allclose(pooled[0], torch.full((3,), (4 + 5 + 6) / 3))
    assert torch.allclose(pooled[1], torch.full((3,), (14 + 15 + 16 + 17) / 4))


def test_grid_cross_check():
    # two images per sample; main images are 1x2x6 (3 merged) and 1x2x8 (4 merged) patches
    grid = torch.tensor([[1, 2, 6], [1, 2, 4], [1, 2, 8], [1, 2, 8]])
    assert W.main_image_tokens_from_grid(grid, 2).tolist() == [3, 4]


def test_hook_captures_input_of_last_layer():
    policy, _ = wam_policy("B")
    assert policy._wam_cfg.hook_layer == -2
    lm = W.backbone_text_model(policy)
    module = W.hooked_module(policy, -2)
    assert module is lm.layers[1]
    with W.HiddenCapture(module) as cap:
        lm(make_ids())
    assert cap.calls == 1 and torch.equal(cap.value, lm.layer_inputs[-1])
    assert len(module._forward_hooks) == 0, "hook must be removed after the context"
    assert W.hooked_module(policy, "final") is lm
    # inside the WAM forward, too: no hook left behind
    policy(make_batch())
    assert all(len(m._forward_hooks) == 0 for m in lm.modules())


def test_hook_accepts_tuple_outputs():
    layer = nn.Linear(3, 3)
    layer.register_forward_hook(lambda m, i, o: (o, None))  # module now returns a tuple
    with W.HiddenCapture(layer) as cap:
        out = layer(torch.randn(2, 3))
    assert torch.equal(cap.value, out[0])


def test_mask_drops_t_plus_10_past_episode_end():
    table, emb, ep, fr = make_table()
    e = torch.tensor([0, 0, 0, 1, 2, 2, 2])
    f = torch.tensor([0, 1, 2, 0, 0, 19, 20])
    _, valid = table.lookup(e, f, "change")
    # ep0 L=12: t=0,1 valid, t=2 -> 12 out; ep1 L=5: none; ep2 L=30: 19 ok, 20 -> 30 out
    assert valid.tolist() == [True, True, False, False, True, True, False]
    _, valid_c = table.lookup(e, f, "current")
    assert torch.equal(valid, valid_c), "arm C uses the same mask as arm B"
    assert table.stats["n_valid"] == 2 + 0 + 20


def test_lookup_values_change_and_current():
    table, emb, ep, fr = make_table()
    e, f = torch.tensor([2]), torch.tensor([7])
    row = 12 + 5 + 7
    chg, _ = table.lookup(e, f, "change")
    cur, _ = table.lookup(e, f, "current")
    m, s = table.stats["change"]
    want_chg = torch.tensor(emb[row + 10]).float() - torch.tensor(emb[row]).float()
    assert torch.allclose(chg[0], (want_chg - m) / s, atol=1e-5)
    m, s = table.stats["current"]
    assert torch.allclose(cur[0], (torch.tensor(emb[row]).float() - m) / s, atol=1e-5)


def test_lookup_rejects_wrong_dataset():
    table = make_table()[0]
    for e, f in ((5, 0), (1, 5)):
        try:
            table.lookup(torch.tensor([e]), torch.tensor([f]), "change")
        except IndexError:
            continue
        raise AssertionError(f"({e},{f}) should be rejected")


def test_table_rejects_unsorted_rows():
    emb = np.zeros((4, 3), np.float16)
    try:
        W.TargetTable(emb, [0, 0, 1, 1], [1, 0, 0, 1])
    except ValueError:
        return
    raise AssertionError("unsorted frames must be rejected")


def test_arm_a_loss_is_exactly_the_stock_flow_loss():
    batch = make_batch()
    torch.manual_seed(0)
    stock = FakePolicy()
    torch.manual_seed(123)
    stock_loss, _ = stock(batch)
    policy, _ = wam_policy("A", seed=0)
    torch.manual_seed(123)
    loss, out = policy(batch)
    assert torch.equal(loss, stock_loss)
    assert out["flow_loss"] == stock_loss.item() and "future_loss" not in out
    assert out["img_tok_main"] == 3.5 and not hasattr(policy, "wam_head")
    assert len(policy.get_optim_params()) == 2


def test_arm_b_flow_part_unchanged_and_loss_combines():
    batch = make_batch()
    torch.manual_seed(0)
    stock = FakePolicy()
    torch.manual_seed(123)
    stock_loss, _ = stock(batch)
    policy, _ = wam_policy("B", seed=0, weight=0.25)
    torch.manual_seed(123)
    loss, out = policy(batch)
    assert out["flow_loss"] == stock_loss.item(), "head must not change the flow path"
    assert abs(loss.item() - (out["flow_loss"] + 0.25 * out["future_loss"])) < 1e-5


def test_arm_b_and_c_target_selection():
    table, emb, _, _ = make_table()
    batch = make_batch()
    for arm, kind in (("B", "change"), ("C", "current")):
        policy, _ = wam_policy(arm, table=table)
        assert policy._wam_cfg.target_kind == kind
        _, out = policy(batch)  # zero-init head predicts 0 -> loss = mean(target^2) on valid
        target, valid = table.lookup(batch[W.EP_KEY], batch[W.FR_KEY], kind)
        want = (target[valid] ** 2).mean().item()
        assert valid.tolist() == [True, False]
        assert abs(out["future_loss"] - want) < 1e-5, (arm, out["future_loss"], want)
    b_target = table.lookup(torch.tensor([0]), torch.tensor([1]), "change")[0]
    c_target = table.lookup(torch.tensor([0]), torch.tensor([1]), "current")[0]
    assert not torch.allclose(b_target, c_target)


def test_masked_mse_handles_all_invalid():
    pred = torch.randn(3, 4, requires_grad=True)
    loss = W.masked_mse(pred, torch.randn(3, 4), torch.zeros(3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0 and torch.count_nonzero(pred.grad) == 0


def test_saved_state_dict_has_no_head_keys():
    policy, _ = wam_policy("B")
    assert any(k.startswith(W.HEAD_PREFIX) for k in policy.state_dict())
    policy._save_pretrained("unused")
    stock_keys = set(FakePolicy().state_dict())
    assert set(policy.saved) == stock_keys
    # a caller-supplied (e.g. gathered) state dict is filtered too
    policy._save_pretrained("unused", state_dict=policy.state_dict())
    assert not any(k.startswith(W.HEAD_PREFIX) for k in policy.saved)


def _future_only_loss(policy, batch):
    """Just the future part of arm B's loss, to isolate its gradient."""
    lm = W.backbone_text_model(policy)
    with W.HiddenCapture(W.hooked_module(policy, -2)) as cap:
        lm(batch["input_ids"])
    pooled = W.pool_main_camera(cap.value, batch["input_ids"], IMG)[0]
    target, valid = policy._wam_targets.lookup(batch[W.EP_KEY], batch[W.FR_KEY], "change")
    return W.masked_mse(policy.wam_head(pooled), target, valid)


def test_future_grad_reaches_layer_minus2_not_last_layer():
    policy, _ = wam_policy("B", weight=1.0)
    # zero-init fc2 blocks gradient at step 0 (by design); perturb to test the wiring
    nn.init.normal_(policy.wam_head.fc2.weight, std=0.1)
    lm = W.backbone_text_model(policy)
    policy.zero_grad()
    _future_only_loss(policy, make_batch()).backward()
    g = lm.layers[-2].mlp.weight.grad
    assert g is not None and g.abs().sum() > 0, "lesson must reach layers[-2]"
    assert lm.embed.weight.grad.abs().sum() > 0
    g = lm.layers[-1].mlp.weight.grad
    assert g is None or g.abs().sum() == 0, "lesson must NOT reach the last layer's MLP"
    assert lm.norm.weight.grad is None and policy.expert.weight.grad is None


def test_combined_arm_b_backward():
    policy, _ = wam_policy("B", weight=1.0)
    nn.init.normal_(policy.wam_head.fc2.weight, std=0.1)
    policy.zero_grad()
    loss, _ = policy(make_batch())
    loss.backward()
    lm = W.backbone_text_model(policy)
    assert lm.layers[-2].mlp.weight.grad.abs().sum() > 0
    assert policy.wam_head.fc1.weight.grad.abs().sum() > 0
    assert lm.layers[-1].mlp.weight.grad is None  # neither flow nor future reads its output


def test_arm_a_backbone_last_layer_gets_no_grad():
    policy, _ = wam_policy("A")
    loss, _ = policy(make_batch())
    loss.backward()
    lm = W.backbone_text_model(policy)
    assert lm.layers[-1].mlp.weight.grad is None
    assert lm.layers[-2].mlp.weight.grad.abs().sum() > 0  # via the last layer's input (K/V)


def test_optimizer_groups_add_head():
    policy, _ = wam_policy("B")
    groups = policy.get_optim_params()
    assert len(groups) == 3 and groups[2]["lr"] == 1e-4
    head_ids = {id(p) for p in policy.wam_head.parameters()}
    assert {id(p) for p in groups[2]["params"]} == head_ids
    assert not head_ids & {id(p) for g in groups[:2] for p in g["params"]}


def test_attach_does_not_touch_global_rng():
    torch.manual_seed(5)
    policy = FakePolicy()
    state = torch.get_rng_state()
    W.attach_wam(policy, W.WAMConfig(arm="B", head_hidden=24), make_table()[0])
    assert torch.equal(state, torch.get_rng_state())


def test_attach_twice_refused():
    policy, table = wam_policy("B")
    try:
        W.attach_wam(policy, W.WAMConfig(arm="B"), table)
    except RuntimeError:
        return
    raise AssertionError("second attach must fail")


def test_carry_index_keeps_episode_and_frame():
    class Pre:
        saved_to = None

        def __call__(self, batch):
            return {"input_ids": batch["x"]}

        def save_pretrained(self, d):
            self.saved_to = d

    pre = W.CarryIndex(Pre())
    out = pre({"x": 1, "episode_index": torch.tensor([[3], [4]]),
               "frame_index": torch.tensor([5, 6])})
    assert out[W.EP_KEY].tolist() == [3, 4] and out[W.FR_KEY].tolist() == [5, 6]
    pre.save_pretrained("d")
    assert pre._inner.saved_to == "d"


def test_wam_config_json_records_hook_layer():
    import json
    with tempfile.TemporaryDirectory() as d:
        W.WAMConfig(arm="B", hook_layer=-3).save(d)
        saved = json.load(open(os.path.join(d, W.WAM_CONFIG_FILE)))
    assert saved["hook_layer"] == -3 and W.WAMConfig().hook_layer == -2


def test_table_load_roundtrip():
    table, emb, ep, fr = make_table()
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "vjepa_targets.npy"), emb)
        np.savez(os.path.join(d, "vjepa_index.npz"), episode_index=ep, frame_index=fr)
        loaded = W.TargetTable.load(d)
    assert torch.equal(loaded.emb, table.emb)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
