"""Check eval_delay.chunk_rollout's schedule without LeRobot, a simulator or a GPU.

Stubs the few LeRobot names the script imports, then drives chunk_rollout with a fake env and a fake
policy whose action k of the chunk predicted at step t is the pair (t, k). The test checks, for every
executed step s, which chunk and which index the robot used:
  s < D + n          -> chunk born at 0, index s
  otherwise          -> chunk born at t = n * floor((s - D) / n), index s - t
Run: python3 -I test_delay_schedule.py
"""

import os
import sys
import types

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


for pkg in ["lerobot", "lerobot.scripts", "lerobot.envs", "lerobot.utils"]:
    _stub(pkg)
_stub("lerobot.scripts.lerobot_eval", rollout=None, main=lambda: None)
_stub(
    "lerobot.envs.utils",
    check_env_attributes_and_types=lambda env: None,
    preprocess_observation=lambda obs: {k: torch.as_tensor(v) for k, v in obs.items()},
)
_stub("lerobot.utils.constants", ACTION="action", OBS_STR="observation")
_stub("lerobot.utils.utils", inside_slurm=lambda: True)

N_STEPS, STRIDE, HORIZON = 100, 10, 50


class FakeEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.t = 0
        self.executed = []

    def call(self, name):
        if name == "_max_episode_steps":
            return [N_STEPS] * self.num_envs
        if name == "task_description":
            return ["task"] * self.num_envs
        raise AttributeError(name)

    def _obs(self):
        return {"t": np.full((self.num_envs, 1), self.t, dtype=np.float32)}

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.executed.append(action[0].copy())
        self.t += 1
        zeros = np.zeros(self.num_envs, dtype=bool)
        return self._obs(), np.zeros(self.num_envs), zeros, zeros, {}


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(n_action_steps=STRIDE)

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        t = batch["t"][:, 0]
        b = t.shape[0]
        idx = torch.arange(HORIZON, dtype=torch.float32).expand(b, HORIZON)
        return torch.stack([t[:, None].expand(b, HORIZON), idx], dim=-1)


def expected(s, d):
    if s < d + STRIDE:
        return (0, s)
    born = STRIDE * ((s - d) // STRIDE)
    return (born, s - born)


def run(d, num_envs=1):
    os.environ["GRIFFIN_DELAY"] = str(d)
    sys.modules.pop("eval_delay", None)
    import eval_delay

    env = FakeEnv(num_envs)
    ident = lambda x: x  # noqa: E731
    eval_delay.chunk_rollout(env, FakePolicy(), ident, ident, ident, ident)
    got = [(int(a[0]), int(a[1])) for a in env.executed]
    want = [expected(s, d) for s in range(N_STEPS)]
    assert got == want, f"delay {d}: first mismatch at step {next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)}"
    return got


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    for d in [0, 2, 5, 10, 20, 40]:
        got = run(d)
        print(f"delay {d:>2}: OK  steps 0-12 used (chunk born, index): {got[:3]} ... step 25 -> {got[25]}")
    run(5, num_envs=2)
    print("batch of 2 envs: OK")
    try:
        run(41)
    except SystemExit as e:
        print("delay 41 refused as expected:", e)
