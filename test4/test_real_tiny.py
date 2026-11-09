"""Same checks as test_wam.py, but on the REAL GriffinAlphaPolicy with a tiny random Qwen3-VL.

Needs the Alpha-S environment (lerobot 0.6.1 + lerobot_policy_griffin_alpha + transformers
5.5.x); no weights or downloads. Runs on CPU in about a minute:

    python test_real_tiny.py

Built like alpha-s tests/test_modeling_griffin_alpha_policy.py (tiny text 256 wide, head_dim 128,
8 kv heads), but with 3 text layers (so layers[-2] is not the first layer), TWO images per
sample (main 4x4 patches -> 4 tokens, wrist 2x2 -> 1 token) and a left-padded row, so the
main-camera pooling is exercised on a real prompt layout.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.models.qwen3_vl import Qwen3VLConfig  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from lerobot_policy_griffin_alpha import GriffinAlphaConfig, GriffinAlphaPolicy  # noqa: E402

import wam_head as W  # noqa: E402

IMG = 5
CHUNK = 4
ADIM = 7


def tiny_policy() -> GriffinAlphaPolicy:
    text = Qwen3VLTextConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                             num_attention_heads=8, num_key_value_heads=8, head_dim=128,
                             vocab_size=200)
    vision = Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128, depth=2, num_heads=2,
                                 patch_size=16, spatial_merge_size=2, out_hidden_size=256,
                                 deepstack_visual_indexes=[0])
    qwen = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    qwen.image_token_id = IMG
    cfg = GriffinAlphaConfig(qwen3vl_config=qwen, horizon=CHUNK, n_action_steps=CHUNK,
                             expert_width=64, expert_mlp_dim=128, expert_depth=3,
                             expert_num_attention_heads=8, action_dim_override=ADIM,
                             attn_implementation="sdpa", gradient_checkpointing=False,
                             device="cpu")
    torch.manual_seed(0)
    return GriffinAlphaPolicy(cfg).to(torch.float32)


def tiny_batch() -> dict:
    v0, v1 = 151, 152  # stand-ins for <|vision_start|>, <|vision_end|>
    row0 = [0, 0, 10, v0] + [IMG] * 4 + [v1, v0, IMG, v1, 11, 12]  # left-padded
    row1 = [10, 13, v0] + [IMG] * 4 + [v1, v0, IMG, v1, 11, 12, 14]
    ids = torch.tensor([row0, row1])
    attn = torch.ones_like(ids)
    attn[0, :2] = 0
    mm = (ids == IMG).to(torch.int32)
    grid = torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 4], [1, 2, 2]])
    n_patches = int(grid.prod(1).sum())
    return {
        "input_ids": ids, "attention_mask": attn, "mm_token_type_ids": mm,
        "pixel_values": torch.randn(n_patches, 3 * 2 * 16 * 16), "image_grid_thw": grid,
        "action": torch.randn(2, CHUNK, ADIM),
        W.EP_KEY: torch.tensor([0, 1]), W.FR_KEY: torch.tensor([3, 2]),
    }


def tiny_table(dim=16):
    rng = np.random.default_rng(0)
    lengths = (20, 14)
    emb = rng.standard_normal((sum(lengths), dim)).astype(np.float16)
    ep = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
    fr = np.concatenate([np.arange(n) for n in lengths])
    return W.TargetTable(emb, ep, fr)


def wam(arm, weight=0.1):
    policy = tiny_policy()
    cfg = W.WAMConfig(arm=arm, future_weight=weight, head_hidden=32)
    return W.attach_wam(policy, cfg, None if arm == "A" else tiny_table())


def test_hook_is_the_input_of_the_last_decoder_layer():
    policy = wam("B")
    batch = tiny_batch()
    lm = W.backbone_text_model(policy)
    assert type(lm).__name__ == "Qwen3VLTextModel", type(lm).__name__
    module = W.hooked_module(policy, policy._wam_cfg.hook_layer)
    assert module is lm.layers[1] and len(lm.layers) == 3
    inputs = policy._filter_model_inputs(batch)
    seen = {}
    pre = lm.layers[-1].register_forward_pre_hook(
        lambda m, args, kwargs: seen.__setitem__("x", args[0] if args else
                                                 kwargs["hidden_states"]), with_kwargs=True)
    policy.eval()
    try:
        with torch.no_grad(), W.HiddenCapture(module) as cap:
            policy._run_prefix(inputs, with_grad=False)
    finally:
        pre.remove()
    assert cap.calls == 1 and cap.value.shape == (2, 14, 256)
    assert torch.equal(cap.value, seen["x"]), "layers[-2] output must equal last layer input"


def test_pooling_matches_grid_on_real_prompt():
    batch = tiny_batch()
    mask = W.first_image_run_mask(batch["input_ids"], IMG)
    want = W.main_image_tokens_from_grid(batch["image_grid_thw"], 2)
    assert mask.sum(1).tolist() == want.tolist() == [4, 4]


def test_arm_a_equals_stock():
    batch = tiny_batch()
    stock = tiny_policy()
    torch.manual_seed(7)
    stock_loss, _ = stock.forward(batch)
    policy = wam("A")
    torch.manual_seed(7)
    loss, out = policy.forward(batch)
    assert torch.equal(loss, stock_loss), (loss, stock_loss)
    assert out["img_tok_main"] == 4.0 and "future_loss" not in out


def test_arm_b_flow_unchanged():
    batch = tiny_batch()
    stock = tiny_policy()
    torch.nn.init.normal_(stock.expert.action_out_proj.weight, std=0.02)
    torch.manual_seed(7)
    stock_loss, _ = stock.forward(batch)
    stock_loss.backward()
    last = stock.model.model.language_model.layers[-1].mlp.down_proj.weight
    assert last.grad is None, "stock flow loss should not reach the last layer's MLP"
    policy = wam("B", weight=1.0)
    torch.nn.init.normal_(policy.expert.action_out_proj.weight, std=0.02)
    torch.nn.init.normal_(policy.wam_head.fc2.weight, std=0.1)
    torch.manual_seed(7)
    _, out = policy.forward(batch)
    assert abs(out["flow_loss"] - stock_loss.item()) < 1e-6, (out, stock_loss)
    assert out["future_valid"] == 1.0


def test_future_grad_reaches_layer_minus2_not_last_layer():
    """The expert's action_out_proj stays zero-initialised, so the flow loss sends NO gradient
    into the backbone: every backbone gradient below comes from the future loss alone."""
    policy = wam("B", weight=1.0)
    assert torch.count_nonzero(policy.expert.action_out_proj.weight) == 0
    torch.nn.init.normal_(policy.wam_head.fc2.weight, std=0.1)
    loss, _ = policy.forward(tiny_batch())
    loss.backward()
    lm = policy.model.model.language_model
    g = lm.layers[-2].mlp.down_proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "lesson must reach layers[-2]"
    g = lm.layers[-1].mlp.down_proj.weight.grad
    assert g is None or g.abs().sum() == 0, "lesson must NOT reach layers[-1].mlp"
    assert lm.norm.weight.grad is None or lm.norm.weight.grad.abs().sum() == 0
    vis = [p for n, p in policy.model.named_parameters() if "visual" in n and p.grad is not None]
    assert vis and sum(float(p.grad.abs().sum()) for p in vis) > 0


def test_saved_checkpoint_is_plain_and_loads_strictly():
    policy = wam("C")
    stock_keys = set(tiny_policy().state_dict())
    with tempfile.TemporaryDirectory() as d:
        policy.save_pretrained(d)
        with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
            keys = set(f.keys())
        assert not any(k.startswith(W.HEAD_PREFIX) for k in keys)
        assert keys <= stock_keys and len(keys) >= len(stock_keys) - 1  # tied lm_head dedup
        loaded = GriffinAlphaPolicy.from_pretrained(d, strict=True)
        assert type(loaded) is GriffinAlphaPolicy and not hasattr(loaded, "wam_head")
        for k, v in loaded.state_dict().items():
            assert torch.equal(v, policy.state_dict()[k].to(v.dtype)), k


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
