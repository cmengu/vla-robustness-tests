# Paired breakdown of Test 3 by LIBERO-Plus perturbation factor.
# Reads ~/t3names.json and every eval_info.json under the two models' t3 output folders.
import glob, json, math, os, re, sys
from collections import defaultdict
H = os.path.expanduser("~")
names = json.load(open(f"{H}/t3names.json"))
name_of = {(s, i): n for s, rows in names.items() for i, n, _, _ in rows}
DEFAULT_VIEW = "0_0_100_0_0"  # LIBERO-Plus's unperturbed camera; initstate_0 = default pose
def factor(name):
    f = []
    if "_language_" in name: f.append("language")
    if "_light_" in name: f.append("lighting")
    if re.search(r"_(?:table|tb)_\d+", name): f.append("layout: table")
    if re.search(r"_add_|_level", name): f.append("layout: extra objects")
    if "_noise_" in name: f.append("sensor noise")
    m = re.search(r"_view_(\d+_\d+_\d+_\d+_\d+)", name)
    if m and m.group(1) != DEFAULT_VIEW: f.append("camera view")
    m = re.search(r"_initstate_(\d+)", name)
    if m and m.group(1) != "0": f.append("robot start pose")
    return "+".join(f) if f else "original/unmatched"
def load(pattern):
    res = {}
    for p in glob.glob(pattern):
        info = json.load(open(p))
        for t in info.get("per_task", []):
            m = t["metrics"]; succ = m.get("pc_success")
            if succ is None and "successes" in m:
                succ = 100.0 * sum(m["successes"]) / len(m["successes"])
            res[(t["task_group"], int(t["task_id"]))] = succ >= 50.0
    return res
A = load(f"{H}/alpha-s/outputs/t3/alphas-*/eval_info.json")
F = load(f"{H}/lerobot/outputs/t3/fastwam-*/eval_info.json")
both = sorted(set(A) & set(F))
print(f"alpha-s cases {len(A)}, fast-wam cases {len(F)}, paired {len(both)}")
def pbinom_two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
def table(keyfn, title):
    g = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, a_succ, f_succ, b (A ok F fail), c (A fail F ok)
    for k in both:
        r = g[keyfn(k)]; a, f = A[k], F[k]
        r[0] += 1; r[1] += a; r[2] += f; r[3] += a and not f; r[4] += f and not a
    print(f"\n== {title}")
    hdr = ("group", "n", "Alpha-S%", "FastWAM%", "gap", "A>F", "F>A", "p")
    print("%-28s %4s %9s %9s %6s %4s %4s %7s" % hdr)
    for key, (n, a, f, b, c) in sorted(g.items(), key=lambda kv: -kv[1][0]):
        print("%-28s %4d %9.1f %9.1f %6.1f %4d %4d %7.4f" % (
            key, n, 100*a/n, 100*f/n, 100*(a-f)/n, b, c, pbinom_two_sided(b, c)))
table(lambda k: factor(name_of[k]), "by perturbation factor (all suites)")
table(lambda k: k[0], "by suite")
table(lambda k: f"{k[0]} / {factor(name_of[k])}", "by suite and factor")
unm = [name_of[k] for k in both if factor(name_of[k]) == "original/unmatched"][:12]
print("\nunmatched name examples:", *unm, sep="\n  ")
