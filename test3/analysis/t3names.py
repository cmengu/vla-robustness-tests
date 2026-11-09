# Dump LIBERO-Plus task names per suite, in task_id order, to ~/t3names.json.
import json, os
from libero.libero import benchmark
b = benchmark.get_benchmark_dict()
out = {}
for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    suite = b[s]()
    out[s] = [[i, t.name, t.bddl_file, t.language] for i, t in enumerate(suite.tasks)]
with open(os.path.expanduser("~/t3names.json"), "w") as f:
    json.dump(out, f)
print({s: len(v) for s, v in out.items()})
