"""Probe for tt-metal #58082: test_softmax_with_3D sporadic wrong output on wh_n150.

  python probe58082.py make            write the probe copy of test_softmax.py into the tree (run from TT_METAL_HOME)
  python probe58082.py report <dir>    summarise stats.txt and hits.jsonl of a run
"""
import json
import os
import re
import sys

TEST = "tests/ttnn/unit_tests/operations/fused/test_softmax.py"
PROBE = "tests/ttnn/unit_tests/operations/fused/test_softmax_probe58082.py"
# A passing run stays within the test's atol + rtol * ref, about 0.012; the CI failures show about 1.0.
BAD = 0.05

BODY = '''@pytest.mark.merge_gate
def test_softmax_with_3D(device):
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe58082", os.environ["PROBE_PY"])
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.run(device, ttnn, torch_random, TEST_PADDING_VALUE)'''

BLOCK = re.compile(
    r"(?:# Not in the Merge Gate[^\n]*\n)?(?:(?:# )?@pytest\.mark\.merge_gate\n)?"
    r"def test_softmax_with_3D\(device\):\n.*?(?=\n\n\n(?:def |@pytest))",
    re.S,
)


def make():
    src = open(TEST).read()
    new, n = BLOCK.subn(lambda _: BODY, src)
    if n != 1:
        sys.exit(f"test_softmax_with_3D block matched {n} times in {TEST}")
    open(PROBE, "w").write(new)
    print(f"wrote {PROBE}")


def split(units, cores):
    """split_work_to_cores: the first units % cores cores take one extra unit."""
    q, rem = divmod(units, cores)
    out, start = [], 0
    for k in range(cores):
        n = q + (1 if k < rem else 0)
        out.append((start, n))
        start += n
    return out


def analyse(got, ref, x, grid_x, grid_y):
    import torch

    ht = (got.shape[1] + 31) // 32
    per_core = split(got.shape[0] * ht, grid_x * grid_y)
    d = torch.nan_to_num((got - ref).abs(), nan=float("inf"))
    bad = d > BAD
    rows = bad.any(dim=-1).nonzero().tolist()
    out = {
        "max_abs_diff": float(d.max()),
        "bad_elements": int(bad.sum()),
        "bad_rows": len(rows),
        "nan": int(torch.isnan(got).sum()),
        "rows": [],
    }
    for b, r in rows[:80]:
        g = torch.nan_to_num(got[b, r], nan=float("inf"))
        cols = bad[b, r].nonzero().flatten().tolist()
        tile_row = b * ht + r // 32
        core = next(k for k, (s, n) in enumerate(per_core) if s <= tile_row < s + n)
        c = int(g.argmax())
        out["rows"].append(
            {
                "b": b,
                "r": r,
                "tile_row": tile_row,
                "row_in_tile": r % 32,
                "core": core,
                "core_xy": [core % grid_x, core // grid_x],
                "pos_in_core": tile_row - per_core[core][0],
                "core_rows": per_core[core][1],
                "n_bad": len(cols),
                "n_gt_0p002": int((d[b, r] > 0.002).sum()),
                "bad_cols": cols[:10],
                "bad_tile_cols": sorted({k // 32 for k in cols})[:12],
                "got_argmax": c,
                "got_max": float(g[c]),
                "got_sum": float(got[b, r].sum()),
                "ref_at_argmax": float(ref[b, r, c]),
                "ref_argmax": int(ref[b, r].argmax()),
                "x_at_argmax": float(x[b, r, c]),
                "x_max": float(x[b, r].max()),
            }
        )
    return out


def brief(got, ref):
    import torch

    d = torch.nan_to_num((got - ref).abs(), nan=float("inf"))
    return {"max_abs_diff": float(d.max()), "bad_rows": (d > BAD).any(dim=-1).nonzero().tolist()[:40]}


def run(device, ttnn, torch_random, pad_value):
    import torch
    import torch.nn.functional as F

    out_dir = os.environ["PROBE_OUT"]
    loop = int(os.environ.get("PROBE_LOOP", "0"))
    reps = int(os.environ.get("PROBE_REPS", "4"))
    grid = device.compute_with_storage_grid_size()

    torch.manual_seed(0)
    x = torch_random((8, 1500, 1500), -10, 10, dtype=torch.bfloat16)
    ref = F.softmax(x, dim=-1, dtype=torch.bfloat16).float()

    def upload():
        t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.fill_implicit_tile_padding(t, pad_value)

    def softmax_of(t):
        return ttnn.to_torch(ttnn.from_device(ttnn.softmax(t, dim=-1))).float()

    fired = 0
    for rep in range(reps):
        t = upload()
        got = softmax_of(t)
        mx = float(torch.nan_to_num((got - ref).abs(), nan=float("inf")).max())
        hit = mx > BAD
        with open(f"{out_dir}/stats.txt", "a") as f:
            f.write(f"{loop} {rep} {mx:.6f} {int(hit)}\n")
        if not hit:
            continue
        fired += 1
        rec = {"loop": loop, "rep": rep, "grid": [grid.x, grid.y]}
        rec.update(analyse(got, ref, x.float(), grid.x, grid.y))
        back = ttnn.to_torch(ttnn.from_device(t)).float()
        rec["input_readback_mismatches"] = int((back != x.float()).sum())
        rec["rerun_same_input"] = []
        for _ in range(3):
            g2 = softmax_of(t)
            rec["rerun_same_input"].append(dict(brief(g2, ref), identical_to_hit=bool(torch.equal(g2, got))))
        rec["rerun_fresh_input"] = brief(softmax_of(upload()), ref)
        if len([p for p in os.listdir(out_dir) if p.endswith(".pt")]) < 3:
            torch.save(got.to(torch.bfloat16), f"{out_dir}/hit_loop{loop}_rep{rep}.pt")
        with open(f"{out_dir}/hits.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    assert fired == 0, f"{fired} corrupted softmax run(s) in loop {loop}, see {out_dir}/hits.jsonl"


def report(out_dir):
    stats = [l.split() for l in open(f"{out_dir}/stats.txt")] if os.path.exists(f"{out_dir}/stats.txt") else []
    clean = [float(s[2]) for s in stats if s[3] == "0"]
    first = [s for s in stats if s[1] == "0"]
    print(f"softmax runs: {len(stats)} over {len({s[0] for s in stats})} loops; first-in-session runs: {len(first)}")
    if clean:
        print(f"clean runs: max abs diff between {min(clean):.6f} and {max(clean):.6f}")
    hits = [json.loads(l) for l in open(f"{out_dir}/hits.jsonl")] if os.path.exists(f"{out_dir}/hits.jsonl") else []
    print(f"hits: {len(hits)} (first-in-session: {sum(h['rep'] == 0 for h in hits)})")
    for h in hits:
        print(
            f"--- loop {h['loop']} rep {h['rep']} grid {h['grid']}: max_abs_diff {h['max_abs_diff']:.4f}, "
            f"{h['bad_elements']} bad elements in {h['bad_rows']} rows, nan {h['nan']}, "
            f"input readback mismatches {h['input_readback_mismatches']}"
        )
        print("    rerun same input:", [(round(r["max_abs_diff"], 4), len(r["bad_rows"]), r["identical_to_hit"]) for r in h["rerun_same_input"]])
        print("    rerun fresh input:", round(h["rerun_fresh_input"]["max_abs_diff"], 4), len(h["rerun_fresh_input"]["bad_rows"]))
        for r in h["rows"][:40]:
            print(
                f"    b{r['b']} r{r['r']} tile_row {r['tile_row']} (row {r['row_in_tile']} of tile) core {r['core']} {r['core_xy']} "
                f"pos {r['pos_in_core']}/{r['core_rows']} | n_bad {r['n_bad']} (over 0.002: {r['n_gt_0p002']}) tile_cols {r['bad_tile_cols']} | "
                f"got max {r['got_max']:.4f} at col {r['got_argmax']} (ref {r['ref_at_argmax']:.2e}, x {r['x_at_argmax']:.3f}, "
                f"row x_max {r['x_max']:.3f}, ref argmax {r['ref_argmax']}) sum {r['got_sum']:.4f}"
            )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make()
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        sys.exit(__doc__)
