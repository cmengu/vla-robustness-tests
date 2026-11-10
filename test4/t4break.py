# Test 4 paired comparison: arms A/B/C vs each other and vs stock Alpha-S, by factor and suite.
# Reads ~/t3names.json and eval_info.json under ~/alpha-s/outputs/t3/{alphas,armA,armB,armC}-*.
import glob, json, math, os, re
from collections import defaultdict
H = os.path.expanduser("~")
names = json.load(open(f"{H}/t3names.json"))
name_of = {(s, i): n for s, rows in names.items() for i, n, _, _ in rows}
DEFAULT_VIEW = "0_0_100_0_0"
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
GEOM = ("camera view", "robot start pose", "layout: table")
def load(tag):
    res = {}
    for p in glob.glob(f"{H}/alpha-s/outputs/t3/{tag}-*/eval_info.json"):
        for t in json.load(open(p)).get("per_task", []):
            m = t["metrics"]; s = m.get("pc_success")
            if s is None and "successes" in m: s = 100.0 * sum(m["successes"]) / len(m["successes"])
            res[(t["task_group"], int(t["task_id"]))] = s >= 50.0
    return res
R = {k: load(k) for k in ("alphas", "armA", "armB", "armC", "armA2", "armB2")}
print("cases:", {k: len(v) for k, v in R.items()})
def p2(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def table(X, Y, lx, ly, keyfn, title):
    both = sorted(set(X) & set(Y))
    g = defaultdict(lambda: [0, 0, 0, 0, 0])
    for k in both:
        r = g[keyfn(k)]; x, y = X[k], Y[k]
        r[0] += 1; r[1] += x; r[2] += y; r[3] += x and not y; r[4] += y and not x
    print(f"\n== {lx} vs {ly}: {title} (paired {len(both)})")
    print("%-28s %4s %7s %7s %6s %5s %5s %7s" % ("group", "n", lx+"%", ly+"%", "gap", lx+">", ly+">", "p"))
    for key, (n, x, y, b, c) in sorted(g.items(), key=lambda kv: -kv[1][0]):
        print("%-28s %4d %7.1f %7.1f %6.1f %5d %5d %7.4f" % (key, n, 100*x/n, 100*y/n, 100*(x-y)/n, b, c, p2(b, c)))
def geom(k): return "geometry (camera/pose/table)" if factor(name_of[k]) in GEOM else "appearance/other"
PAIRS = (("armB", "armA"), ("armC", "armA"), ("armB", "armC"), ("armA", "alphas"),
         ("armB2", "armB"), ("armA2", "armA"), ("armB2", "armA"), ("armB2", "armA2"))
for lx, ly in PAIRS:
    X, Y = R[lx], R[ly]
    if not X or not Y: print(f"\n== {lx} vs {ly}: skipped (no cases yet)"); continue
    table(X, Y, lx, ly, lambda k: "all", "overall")
    table(X, Y, lx, ly, geom, "geometry vs appearance")
    table(X, Y, lx, ly, lambda k: factor(name_of[k]), "by factor")
    table(X, Y, lx, ly, lambda k: k[0], "by suite")
