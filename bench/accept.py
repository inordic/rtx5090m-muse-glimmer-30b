import json, sys, collections
label = sys.argv[1]
row = None
for l in open("/root/bench/results.jsonl"):
    r = json.loads(l)
    if r.get("label") == label: row = r
if not row: print("  (no row)"); sys.exit()
agg = collections.defaultdict(lambda: [0, 0, 0])
for x in row.get("reps", []):
    if x.get("draft_n"):
        a = agg[x["kind"] + str(x["n"])]; a[0] += x["draft_acc"] or 0; a[1] += x["draft_n"]; a[2] += x["tn"] or 0
print("  acceptance: " + "  ".join(f"{k} {100*a/d:.1f}% ({t} tok)" for k, (a, d, t) in sorted(agg.items())) if agg else "  acceptance: no drafting")
