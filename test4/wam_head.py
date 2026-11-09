"""Test 4: a "future lesson" head on top of Griffin Alpha-S (world-action model, small).

What this module adds to a stock GriffinAlphaPolicy, without editing the installed package:

1. FutureHead: parameter-free LayerNorm, then a 2-layer MLP 2560 -> 2048 -> 1024 (~7.3M params).
2. Hidden-state capture: a forward hook on ``language_model.layers[hook_layer]`` (default -2,
   i.e. decoder layer 34 of 36), whose output is the INPUT of the last decoder layer.
   Why there: the action expert reads each layer's K/V, and layer L's K/V are computed from
   layer L's input. The last layer's attention-out/MLP and the final norm feed nothing the expert
   reads, so a lesson learned only there could not help the actions. Hooking the input of the
   last layer puts the lesson in a tensor the expert does use (through the last layer's K/V).
   This tensor is a plain residual stream: Qwen3-VL adds its "deepstack" visual features only
   to the outputs of the first len(deepstack_visual_embeds) layers (3 for the 4B model), see
   transformers 5.5.4 models/qwen3_vl/modeling_qwen3_vl.py:915-932, and the decoder layer
   returns a bare tensor (same file :564). It is PRE-norm (Qwen has large outlier dimensions),
   hence the parameter-free LayerNorm at the head input. The hook only READS the tensor, so
   the flow-loss path is untouched; it is removed after every forward.
3. Image-token pooling: mean over the token positions of the MAIN camera only. Every image in a
   Qwen3-VL prompt is ``<|vision_start|> <|image_pad|> x n <|vision_end|>``, and the main camera
   (observation.images.image) is the first image in the prompt, so its tokens are the FIRST
   contiguous run of ``image_token_id`` in each row.
4. Target lookup by (episode_index, frame_index) in a precomputed V-JEPA 2 table, with a mask
   that drops samples whose t+10 is past the episode end (same idea as ``_step_mask``).
5. Loss: flow_loss + w * future_loss. Arm A: flow only (stock numbers). Arm B: target is
   emb(t+10) - emb(t). Arm C: target is emb(t). Targets are standardised per dimension (mean 0,
   std 1 over the valid rows) so that w means the same thing in B and C.
6. Saving: the head is dropped from ``model.safetensors``; the checkpoint is a plain
   griffin_alpha checkpoint. The head is saved next to it for resuming only.

Only torch and numpy are imported at module level so the tests run without lerobot.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

FUTURE_STEPS = 10  # 1 second at the dataset's 10 fps
TARGET_DIM = 1024  # V-JEPA 2 ViT-L hidden size
HEAD_PREFIX = "wam_head."
HEAD_FILE = "wam_head.safetensors"
WAM_CONFIG_FILE = "wam_config.json"
# Keys our preprocessor wrapper copies from the raw dataset batch (the Griffin input step
# rebuilds the batch and would otherwise drop them).
EP_KEY = "wam_episode_index"
FR_KEY = "wam_frame_index"
ARMS = ("A", "B", "C")


# ------------------------------------------------------------------------------------------
# The head
# ------------------------------------------------------------------------------------------
class FutureHead(nn.Module):
    """2-layer MLP. Last layer zero-initialised: at step 0 it predicts 0 and sends no gradient
    into the backbone, so the extra loss is switched on gently (same trick as the expert's
    action_out_proj)."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = TARGET_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        # parameter-free: the hooked residual stream is pre-norm with outlier dimensions
        x = F.layer_norm(x, (x.shape[-1],))
        return self.fc2(F.gelu(self.fc1(x)))


# ------------------------------------------------------------------------------------------
# Pooling over the main camera's image tokens
# ------------------------------------------------------------------------------------------
def first_image_run_mask(input_ids: Tensor, image_token_id: int) -> Tensor:
    """(B, P) bool: True on the first contiguous run of image tokens in each row."""
    is_img = input_ids == image_token_id
    prev = torch.zeros_like(is_img)
    prev[:, 1:] = is_img[:, :-1]
    starts = is_img & ~prev
    run_id = torch.cumsum(starts.long(), dim=1)
    return is_img & (run_id == 1)


def pool_main_camera(hidden: Tensor, input_ids: Tensor, image_token_id: int):
    """Mean of ``hidden`` (B, P, H) over the main camera's tokens -> ((B, H) fp32, counts (B,))."""
    if hidden.shape[:2] != input_ids.shape:
        raise ValueError(f"hidden {tuple(hidden.shape)} vs ids {tuple(input_ids.shape)}")
    mask = first_image_run_mask(input_ids, image_token_id)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("a sample has no image tokens; cannot pool the main camera")
    weights = mask.to(torch.float32) / counts[:, None].to(torch.float32)
    pooled = torch.einsum("bp,bph->bh", weights, hidden.to(torch.float32))
    return pooled, counts


def main_image_tokens_from_grid(image_grid_thw: Tensor, batch_size: int, merge: int = 2):
    """Expected main-camera token count per sample from ``image_grid_thw`` (cross-check only).

    Images are listed sample by sample, so with k images per sample the main camera of sample i
    is row i*k. Each merged token covers merge x merge patches."""
    n_img = image_grid_thw.shape[0]
    if n_img % batch_size != 0:
        return None
    per = n_img // batch_size
    rows = image_grid_thw[::per]
    return rows.prod(dim=1) // (merge * merge)


# ------------------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------------------
class TargetTable:
    """V-JEPA 2 embeddings, one per frame, looked up by (episode_index, frame_index).

    Rows must be grouped by episode with frame_index = 0..L-1 in order (vjepa_targets.py --merge
    writes them that way and this class checks it). Row of (e, f) = ep_start[e] + f.
    """

    def __init__(self, emb, episode_index, frame_index, horizon: int = FUTURE_STEPS):
        emb = torch.as_tensor(np.asarray(emb))
        ep = torch.as_tensor(np.asarray(episode_index), dtype=torch.long)
        fr = torch.as_tensor(np.asarray(frame_index), dtype=torch.long)
        if emb.ndim != 2 or emb.shape[0] != ep.shape[0] or ep.shape != fr.shape:
            raise ValueError("emb (N, D), episode_index (N,), frame_index (N,) do not line up")
        n_ep = int(ep.max()) + 1
        ep_len = torch.bincount(ep, minlength=n_ep)
        ep_start = torch.zeros(n_ep, dtype=torch.long)
        ep_start[1:] = torch.cumsum(ep_len, 0)[:-1]
        expect_ep = torch.repeat_interleave(torch.arange(n_ep), ep_len)
        expect_fr = torch.arange(ep.shape[0]) - ep_start[expect_ep]
        if not (torch.equal(ep, expect_ep) and torch.equal(fr, expect_fr)):
            raise ValueError("rows are not grouped by episode with frames 0..L-1 in order")
        self.horizon = horizon
        self.emb = emb  # fp16, (N, D)
        self.ep_start = ep_start
        self.ep_len = ep_len
        self.dim = emb.shape[1]
        self.stats = self._compute_stats()

    @classmethod
    def load(cls, target_dir: str, horizon: int = FUTURE_STEPS, mmap: bool = True):
        path = os.path.join(target_dir, "vjepa_targets.npy")
        emb = np.load(path, mmap_mode="r" if mmap else None)
        idx = np.load(os.path.join(target_dir, "vjepa_index.npz"))
        return cls(np.array(emb, dtype=np.float16), idx["episode_index"], idx["frame_index"],
                   horizon)

    def _valid_rows(self) -> Tensor:
        """Rows (as current-frame rows) whose t+horizon is inside the episode."""
        n = self.emb.shape[0]
        ep = torch.repeat_interleave(torch.arange(self.ep_len.shape[0]), self.ep_len)
        fr = torch.arange(n) - self.ep_start[ep]
        return torch.nonzero(fr + self.horizon < self.ep_len[ep]).squeeze(1)

    def _compute_stats(self, chunk: int = 65536) -> dict:
        """Per-dimension mean/std of the CHANGE and CURRENT targets over the valid rows."""
        rows = self._valid_rows()
        acc = {k: [torch.zeros(self.dim, dtype=torch.float64) for _ in range(2)]
               for k in ("change", "current")}
        for s in range(0, rows.shape[0], chunk):
            r = rows[s:s + chunk]
            cur = self.emb[r].to(torch.float64)
            chg = self.emb[r + self.horizon].to(torch.float64) - cur
            for k, v in (("change", chg), ("current", cur)):
                acc[k][0] += v.sum(0)
                acc[k][1] += (v * v).sum(0)
        n = max(int(rows.shape[0]), 1)
        out = {}
        for k, (s1, s2) in acc.items():
            mean = s1 / n
            std = (s2 / n - mean * mean).clamp_min(1e-12).sqrt()
            out[k] = (mean.float(), std.float().clamp_min(1e-6))
        out["n_valid"] = n
        return out

    def to(self, device):
        self.emb = self.emb.to(device)
        self.ep_start = self.ep_start.to(device)
        self.ep_len = self.ep_len.to(device)
        self.stats = {k: (v if isinstance(v, int) else (v[0].to(device), v[1].to(device)))
                      for k, v in self.stats.items()}
        return self

    def lookup(self, episode_index: Tensor, frame_index: Tensor, kind: str):
        """-> (target (B, D) fp32 standardised, valid (B,) bool). kind: 'change' or 'current'.

        valid is False when t+horizon is past the episode end; for those rows the target is a
        placeholder (zeros) and the loss must ignore them. 'current' uses the SAME mask so arms B
        and C train the head on exactly the same samples.
        """
        dev = self.emb.device
        ep = episode_index.reshape(-1).to(dev, torch.long)
        fr = frame_index.reshape(-1).to(dev, torch.long)
        n_ep = self.ep_len.shape[0]
        if bool(((ep < 0) | (ep >= n_ep)).any()):
            raise IndexError("episode_index outside the target table: wrong dataset?")
        length = self.ep_len[ep]
        if bool(((fr < 0) | (fr >= length)).any()):
            raise IndexError("frame_index outside its episode in the target table: wrong dataset?")
        valid = fr + self.horizon < length
        row = self.ep_start[ep] + fr
        cur = self.emb[row].to(torch.float32)
        if kind == "current":
            target = cur
        elif kind == "change":
            fut_row = torch.where(valid, row + self.horizon, row)
            target = self.emb[fut_row].to(torch.float32) - cur
        else:
            raise ValueError(f"unknown target kind {kind!r}")
        mean, std = self.stats[kind]
        target = (target - mean) / std
        target = torch.where(valid[:, None], target, torch.zeros_like(target))
        return target, valid


# ------------------------------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------------------------------
def masked_mse(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Mean squared error over valid rows (mean over dims, then over kept rows)."""
    per_row = ((pred.float() - target.float()) ** 2).mean(dim=-1)
    keep = valid.to(per_row.dtype)
    return (per_row * keep).sum() / keep.sum().clamp_min(1.0)


def masked_cosine(pred: Tensor, target: Tensor, valid: Tensor) -> float:
    if not bool(valid.any()):
        return float("nan")
    cos = F.cosine_similarity(pred.float()[valid], target.float()[valid], dim=-1)
    return float(cos.mean())


def combine(flow_loss: Tensor, future_loss: Tensor | None, weight: float) -> Tensor:
    """Arm A passes future_loss=None and gets the flow loss tensor back unchanged."""
    if future_loss is None:
        return flow_loss
    return flow_loss + weight * future_loss


# ------------------------------------------------------------------------------------------
# Hidden-state capture
# ------------------------------------------------------------------------------------------
def backbone_text_model(policy) -> nn.Module:
    """GriffinAlphaPolicy.model is Qwen3VLForConditionalGeneration; .model is Qwen3VLModel;
    .language_model is Qwen3VLTextModel, whose forward ends with ``self.norm(hidden_states)``."""
    return policy.model.model.language_model


def hooked_module(policy, layer) -> nn.Module:
    """``layer`` = "final" -> the text model (post-norm output); int k -> layers[k] output."""
    lm = backbone_text_model(policy)
    if layer == "final":
        return lm
    return lm.layers[int(layer)]


class HiddenCapture:
    """Context manager: records the output tensor of ``module`` (or of one decoder layer)."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.value: Tensor | None = None
        self.calls = 0
        self._handle = None

    def _hook(self, _module, _inputs, output):
        if isinstance(output, Tensor):
            tensor = output
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        else:
            tensor = output[0]
        self.value = tensor
        self.calls += 1

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False


# ------------------------------------------------------------------------------------------
# Wiring onto a policy
# ------------------------------------------------------------------------------------------
@dataclass
class WAMConfig:
    arm: str = "A"
    future_weight: float = 0.1
    head_hidden: int = 2048
    head_lr: float = 1e-4
    horizon: int = FUTURE_STEPS
    head_seed: int = 1234
    hook_layer: int = -2  # output of layers[-2] = input of the last decoder layer

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")

    @property
    def target_kind(self) -> str | None:
        return {"A": None, "B": "change", "C": "current"}[self.arm]

    def save(self, directory) -> None:
        with open(os.path.join(directory, WAM_CONFIG_FILE), "w") as f:
            json.dump(asdict(self), f, indent=2)


def _wam_forward(self, batch):
    """Replaces policy.forward. Arm A: the stock forward, plus two logging numbers."""
    wam: WAMConfig = self._wam_cfg
    ids = batch.get("input_ids")
    extra = {}
    if isinstance(ids, Tensor) and isinstance(batch.get("attention_mask"), Tensor):
        extra["tok_per_sample"] = float(batch["attention_mask"].float().sum(1).mean())
        n_main = first_image_run_mask(ids, self._wam_image_token_id).sum(1)
        extra["img_tok_main"] = float(n_main.float().mean())

    if wam.target_kind is None:
        loss, out = self._wam_base_forward(batch)
        out = dict(out or {})
        out.update(extra)
        return loss, out

    if EP_KEY not in batch or FR_KEY not in batch:
        raise KeyError(f"batch lacks {EP_KEY}/{FR_KEY}; wrap the preprocessor (CarryIndex)")
    with HiddenCapture(hooked_module(self, wam.hook_layer)) as cap:
        flow_loss, out = self._wam_base_forward(batch)
    if cap.calls != 1 or cap.value is None:
        raise RuntimeError(f"layer {wam.hook_layer} hook fired {cap.calls} times (expected 1)")
    pooled, _ = pool_main_camera(cap.value, ids, self._wam_image_token_id)
    target, valid = self._wam_targets.lookup(batch[EP_KEY], batch[FR_KEY], wam.target_kind)
    head = self.wam_head
    with torch.autocast(device_type=pooled.device.type, enabled=False):
        pred = head(pooled.to(head.fc1.weight.dtype))
    future_loss = masked_mse(pred, target.to(pred.device), valid.to(pred.device))
    loss = combine(flow_loss, future_loss, wam.future_weight)
    out = dict(out or {})
    out.update(extra)
    out.update({
        "flow_loss": float(flow_loss.detach()),
        "future_loss": float(future_loss.detach()),
        "future_cos": masked_cosine(pred.detach(), target, valid.to(pred.device)),
        "future_valid": float(valid.float().mean()),
        "loss": float(loss.detach()),
    })
    return loss, out


def _wam_get_optim_params(self):
    groups = self._wam_base_get_optim_params()
    if hasattr(self, "wam_head"):
        groups = list(groups)
        head_params = [p for p in self.wam_head.parameters() if p.requires_grad]
        groups.append({"params": head_params, "lr": self._wam_cfg.head_lr})
    return groups


def strip_head(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.startswith(HEAD_PREFIX)}


def _wam_save_pretrained(self, save_directory, state_dict=None):
    sd = self.state_dict() if state_dict is None else state_dict
    return self._wam_base_save_pretrained(save_directory, state_dict=strip_head(sd))


_SUBCLASSES: dict[type, type] = {}


def _wam_subclass(base: type) -> type:
    """A subclass of the policy's own class whose forward / optimizer groups / save are ours.

    Built at run time so the same code wraps GriffinAlphaPolicy on the cluster and a fake policy
    in the tests. Class-level ``name``/``config_class`` are inherited, so the saved config.json is
    unchanged (type ``griffin_alpha``).
    """
    if base in _SUBCLASSES:
        return _SUBCLASSES[base]
    sub = type(
        f"WAM{base.__name__}",
        (base,),
        {
            "forward": _wam_forward,
            "get_optim_params": _wam_get_optim_params,
            "_save_pretrained": _wam_save_pretrained,
            "_wam_base_forward": base.forward,
            "_wam_base_get_optim_params": base.get_optim_params,
            "_wam_base_save_pretrained": base._save_pretrained,
        },
    )
    _SUBCLASSES[base] = sub
    return sub


def backbone_hidden_size(policy) -> int:
    return int(policy.model.config.text_config.hidden_size)


def image_token_id_of(policy) -> int:
    return int(policy.model.config.image_token_id)


def attach_wam(policy, cfg: WAMConfig, targets: TargetTable | None, head_state: dict | None = None):
    """Turn ``policy`` into the WAM variant in place and return it.

    Arm A gets no head and no targets: only the logging extras. The head is built under a forked
    RNG with its own seed, so the global RNG stream (flow noise, flow time, augmentation) is the
    same in all three arms.
    """
    if type(policy) in _SUBCLASSES.values():
        raise RuntimeError("attach_wam called twice on the same policy")
    policy.__class__ = _wam_subclass(type(policy))
    policy._wam_cfg = cfg
    policy._wam_image_token_id = image_token_id_of(policy)
    if cfg.target_kind is None:
        return policy
    if targets is None:
        raise ValueError(f"arm {cfg.arm} needs the V-JEPA target table")
    ref = next(policy.model.parameters())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.head_seed)
        head = FutureHead(backbone_hidden_size(policy), cfg.head_hidden, targets.dim)
    if head_state is not None:
        head.load_state_dict(head_state)
    policy.wam_head = head.to(device=ref.device, dtype=torch.float32)
    policy._wam_targets = targets.to(ref.device)
    return policy


def head_state_dict(policy) -> dict:
    return {k: v.detach().cpu().contiguous() for k, v in policy.wam_head.state_dict().items()}


class CarryIndex:
    """Wraps the lerobot preprocessor so episode/frame indices survive into the policy batch.

    The Griffin input step returns only the tokenised prompt + action fields; everything else in
    the raw dataset batch (including episode_index / frame_index) is dropped. Every other
    attribute (save_pretrained, steps, ...) is delegated, so lerobot saves the real pipeline.
    """

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, batch):
        out = self._inner(batch)
        if isinstance(batch, dict) and "episode_index" in batch and "frame_index" in batch:
            out[EP_KEY] = torch.as_tensor(batch["episode_index"]).reshape(-1)
            out[FR_KEY] = torch.as_tensor(batch["frame_index"]).reshape(-1)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)
