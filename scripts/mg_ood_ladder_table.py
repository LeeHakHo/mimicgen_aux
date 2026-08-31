"""Summarize an mg_verify_ood_ladder.py log into one ID-vs-OOD row per object.

Usage: python mg_ood_ladder_table.py slurm-<verify_job>.out
"""
import collections
import re
import sys

path = sys.argv[1]
task = obj = role = None
rows = []
for ln in open(path):
    m = re.match(r"^([a-z_0-9]+)   base=", ln)
    if m:
        task = m.group(1)
        continue
    m = re.match(r"^  -- (\S+) \[(\w+)", ln)
    if m:
        obj, role = m.group(1), m.group(2)
        continue
    m = re.match(r"^\s+(x|y|z_rot)\s+base \[\s*([-\d.]+),\s*([-\d.]+)\].*?id \[\s*([-\d.]+),"
                 r"\s*([-\d.]+)\].*?ood \[\s*([-\d.]+),\s*([-\d.]+)\]", ln)
    if m and task:
        rows.append((task, obj, role, m.group(1),
                     (float(m.group(4)), float(m.group(5))),
                     (float(m.group(6)), float(m.group(7)))))

by = collections.OrderedDict()
for t, ob, r, k, i, o in rows:
    by.setdefault((t, ob, r), {})[k] = (i, o)

for (t, ob, r), d in by.items():
    cells = []
    for k in ("x", "y", "z_rot"):
        if k not in d:
            cells.append(f"{k}: n/a")
            continue
        i, o = d[k]
        wi, wo = i[1] - i[0], o[1] - o[0]
        if k == "z_rot":
            cells.append(f"yaw {wi:5.0f} -> {wo:5.0f} deg" + ("  =" if abs(wo - wi) < 0.5 else "   "))
        else:
            tag = "  =" if abs(wo - wi) < 1e-6 else (f" x{wo / wi:.2f}" if wi > 0 else "   ")
            cells.append(f"{k} {wi:.3f} -> {wo:.3f}{tag}")
    print(f"{t:24s} {ob:15s} {r:8s} | " + " | ".join(cells))
