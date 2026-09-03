#!/usr/bin/env python3
"""Check that this machine's stack matches the one the ID90 results were produced on.

Run it on any cluster before trusting a number that will be compared against another cluster's:

    python setup_env/verify_env.py

Exits non-zero if anything load-bearing differs. Every expectation below is from
setup_env/ENVIRONMENT.md; the two texture md5s are the ones the runs actually used, which for
those files is NOT what upstream robosuite ships.
"""
import hashlib, importlib, os, subprocess, sys

PINS = {                       # module -> exact version the results were produced with
    "numpy": "1.26.4",         # load-bearing: 2.2.6 took a baseline from 0.29 to 0.02
    "torch": "2.4.0",          # prefix match, +cu121 build
    "mujoco": "2.3.2",
    "h5py": "3.16.0",
    "scipy": "1.15.3",
    "robosuite": "1.4.1",
}
# The two forks are pinned by COMMIT, not by version string -- their __version__ tracks the
# upstream they forked from and does not move when we commit. Reported, never failed on: compare
# the hashes between two machines by eye.
FORKS = ("mimicgen", "robomimic")
COMMITS = {                    # editable checkout -> commit the results came from
    "robosuite": "b9d8d3de",
    "robosuite_task_zoo": "74eab7f",
}
TEXTURES = {                   # robosuite-relative path -> md5 AS USED (not upstream)
    "models/assets/textures/cereal.png": "94d62bdde54befd0bd8d5e4eeb32fbc8",
    "models/assets/textures/soda.png":   "1a2adfdadc322538fee12d1a9b206675",
}
ENVVARS = {"MUJOCO_GL": "egl", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

bad = []
def check(ok, label, got, want):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:34s} {got}" + ("" if ok else f"   != {want}"))
    if not ok:
        bad.append(label)

print("python")
check(sys.version_info[:2] == (3, 10), "python", ".".join(map(str, sys.version_info[:3])), "3.10.x")

print("\nversions")
mods = {}
for name, want in PINS.items():
    try:
        m = importlib.import_module(name)
        mods[name] = m
        got = getattr(m, "__version__", "?")
        check(str(got).split("+")[0] == want, name, got, want)
    except Exception as e:
        check(False, name, f"import failed: {type(e).__name__}", want)

def commit_of(mod):
    root = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
    r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    b = subprocess.run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True)
    d = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                       capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() or "?", b.stdout.strip() or "?", len(d.splitlines())


print("\nfork checkouts (compare these two lines against the other machine)")
for name in FORKS:
    try:
        m = importlib.import_module(name)
        c, b, dirty = commit_of(m)
        print(f"  info  {name:34s} {b} @ {c}"
              + (f"   ({dirty} uncommitted file(s))" if dirty else "")
              + f"   v{getattr(m, '__version__', '?')}")
    except Exception as e:
        check(False, name, f"import failed: {type(e).__name__}", "importable")

print("\ncheckout commits")
for name, want in COMMITS.items():
    m = mods.get(name)
    if m is None:
        try: m = importlib.import_module(name)
        except Exception: check(False, name, "not importable", want); continue
    root = os.path.dirname(os.path.dirname(os.path.abspath(m.__file__)))
    try:
        got = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        got = ""
    check(got.startswith(want) or want.startswith(got), name, got or "not a git checkout", want)

print("\ntextures (pick_place_d0's cereal box and soda can -- 84x84 observations depend on them)")
rs = mods.get("robosuite")
for rel, want in TEXTURES.items():
    if rs is None:
        check(False, rel.split("/")[-1], "robosuite not importable", want); continue
    p = os.path.join(os.path.dirname(os.path.abspath(rs.__file__)), rel)
    if not os.path.exists(p):
        check(False, rel.split("/")[-1], "missing", want); continue
    got = hashlib.md5(open(p, "rb").read()).hexdigest()
    check(got == want, rel.split("/")[-1], got, want)

print("\nenvironment variables")
for k, want in ENVVARS.items():
    got = os.environ.get(k)
    check(got == want, k, got or "<unset>", want)

print("\ntask registration (robosuite_task_zoo is silently optional -- these two vanish without it)")
try:
    import mimicgen.envs.robosuite.ood_ladder as L
    from robosuite.environments.base import REGISTERED_ENVS
    for t in ("hammer_cleanup_d1", "kitchen_d1"):
        check(L.TASK_LADDER[t][0] in REGISTERED_ENVS, t, L.TASK_LADDER[t][0], "registered")
    check(len(L.TASK_LADDER) == 12, "tasks in ladder", len(L.TASK_LADDER), 12)
except Exception as e:
    check(False, "ladder import", f"{type(e).__name__}: {e}", "importable")

print()
if bad:
    print(f"MISMATCH in {len(bad)}: " + ", ".join(bad))
    print("A difference here is enough to move evaluation numbers; see setup_env/ENVIRONMENT.md.")
    sys.exit(1)
print("stack matches the one the published results were produced on")
