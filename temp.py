import csv
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))
lab = {int(float(r["id"])): (r["label"].strip().upper(), r["notes"]) for r in rd("labels_west.csv")}
for r in rd("key_west.csv"):
    i, s = int(float(r["id"])), int(float(r["stratum"]))
    l, n = lab[i]
    if (s == 1 and l == "0" and "bare" not in n.lower()) or (s > 1 and l == "1"):
        print("stratum", s, "| id", i, "| label", l, "|", n)