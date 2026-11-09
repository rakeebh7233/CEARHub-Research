"""
Score a NEW map (v3) against the EXISTING stratified accuracy sample.

Usage:
    python accuracy_v3.py labels_west.csv key_west.csv stats_west_2013_2023.csv eval_v3_west.csv
    python accuracy_v3.py labels_west.csv key_west.csv stats_west_2013_2023.csv eval_v3_west.csv --bare

labels_west.csv  : your labels (id, label, notes)
key_west.csv     : id -> stratum (the ORIGINAL v2 strata)
stats_west_...   : the ORIGINAL v2 stratum areas (s1_m2 .. s4_m2) - these are the
                   sampling weights, so they must be the v2 file, not the v3 stats
eval_v3_west.csv : v3 map values at each point (id, new, newBare, ...) from MODE 'EVAL'

--bare : definition "impervious OR bare ground". Truth: a 0 with "bare" in notes
         (and not "both") counts as 1. Map: new OR newBare counts as mapped.

Why this works: the points were drawn using v2's strata, but your labels describe
the ground, not any map. Each point still stands for its v2 stratum's share of the
area (the weights), so any map can be scored with the same points:
    precision = sum_h W_h * mean(map & truth)   /  sum_h W_h * mean(map)
    recall    = sum_h W_h * mean(map & truth)   /  sum_h W_h * mean(truth)
(Stehman 2014, stratified sample with strata different from the map classes.)
Precision is less exact than for v2 when v3's red falls in strata other than 1,
because those strata have few points per km2.
"""
import csv
import random
import sys

STRATUM_NAMES = {1: "v2 mapped new", 2: "v2 impervious, not new",
                 3: "v2 near new", 4: "v2 everything else"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def estimates(data, w):
    a_my = a_m = a_y = 0.0
    for h, pts in data.items():
        if not pts:
            continue
        n = float(len(pts))
        a_my += w[h] * sum(m * y for m, y in pts) / n
        a_m += w[h] * sum(m for m, _ in pts) / n
        a_y += w[h] * sum(y for _, y in pts) / n
    prec = a_my / a_m if a_m > 0 else float("nan")
    rec = a_my / a_y if a_y > 0 else float("nan")
    return prec, rec, a_m, a_y


def pct(values, q):
    v = sorted(x for x in values if x == x)  # drop NaN
    if not v:
        return float("nan")
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return v[k]


def main(labels_path, key_path, stats_path, eval_path, bare_mode=False,
         n_boot=2000, seed=2026):
    labels, notes = {}, {}
    for r in read_csv(labels_path):
        pid = int(float(r["id"]))
        labels[pid] = (r.get("label") or "").strip().upper()
        notes[pid] = (r.get("notes") or "").strip().lower()
    key = {int(float(r["id"])): int(float(r["stratum"])) for r in read_csv(key_path)}
    evals = {int(float(r["id"])): r for r in read_csv(eval_path)}
    stats = read_csv(stats_path)[0]

    area = {h: float(stats["s%d_m2" % h]) for h in range(1, 5)}
    total = sum(area.values())
    w = {h: area[h] / total for h in area}

    data = {h: [] for h in range(1, 5)}
    false_ids, missed_ids = [], []
    unsure = missing = 0
    for pid, h in sorted(key.items()):
        lab = labels.get(pid, "")
        note = notes.get(pid, "")
        if bare_mode and lab == "0" and "bare" in note and "both" not in note:
            lab = "1"
        if lab not in ("0", "1"):
            unsure += 1
            continue
        ev = evals.get(pid)
        if ev is None:
            missing += 1
            continue
        m = int(float(ev["new"]))
        if bare_mode:
            m = max(m, int(float(ev["newBare"])))
        y = int(lab)
        data[h].append((m, y))
        if m == 1 and y == 0:
            false_ids.append(pid)
        if m == 0 and y == 1:
            missed_ids.append(pid)

    print("Definition: %s" % ("impervious OR bare ground (--bare)" if bare_mode
                              else "impervious = constructed surfaces only"))
    if missing:
        print("WARNING: %d labeled points have no v3 value (masked?)." % missing)

    prec, rec, a_m, a_y = estimates(data, w)

    rng = random.Random(seed)
    bp, br, bm, ba = [], [], [], []
    for _ in range(n_boot):
        res = {h: [rng.choice(p) for _ in p] if p else [] for h, p in data.items()}
        p_, r_, m_, y_ = estimates(res, w)
        bp.append(p_); br.append(r_); bm.append(m_ * total); ba.append(y_ * total)

    print("\nPer v2 stratum")
    print("%-26s %8s %5s %8s %6s %6s %7s" % ("stratum", "weight", "n", "v3_red",
                                             "hits", "false", "missed"))
    for h in range(1, 5):
        pts = data[h]
        print("%-26s %8.4f %5d %8d %6d %6d %7d" % (
            "%d %s" % (h, STRATUM_NAMES[h]), w[h], len(pts),
            sum(m for m, _ in pts), sum(m * y for m, y in pts),
            sum(m * (1 - y) for m, y in pts), sum((1 - m) * y for m, y in pts)))

    ac = 4046.86
    print("\nPrecision v3: %.3f  [95%% CI %.3f-%.3f]" % (prec, pct(bp, .025), pct(bp, .975)))
    print("Recall v3:    %.3f  [95%% CI %.3f-%.3f]" % (rec, pct(br, .025), pct(br, .975)))
    print("v3 mapped area (estimated from sample): %.0f acres  [%.0f-%.0f]" % (
        a_m * total / ac, pct(bm, .025) / ac, pct(bm, .975) / ac))
    print("   (exact v3 pixel area is in stats_v3_west.csv; the two should be similar)")
    print("Adjusted true area: %.0f acres  [%.0f-%.0f]  (same as v2: truth didn't change)" % (
        a_y * total / ac, pct(ba, .025) / ac, pct(ba, .975) / ac))
    print("\nPoints excluded (U or unlabeled): %d" % unsure)
    print("False red ids (map 1, truth 0): %s" % false_ids)
    print("Missed ids (map 0, truth 1):    %s" % missed_ids)


if __name__ == "__main__":
    args = sys.argv[1:]
    bare_flag = "--bare" in args
    args = [a for a in args if a != "--bare"]
    if len(args) != 4:
        print(__doc__)
        sys.exit(1)
    main(args[0], args[1], args[2], args[3], bare_mode=bare_flag)