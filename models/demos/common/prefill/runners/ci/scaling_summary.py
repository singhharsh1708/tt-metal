# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Compare the disaggregated-prefill runner legs across cluster configs and report the sc4 gain.

Each leg drops a metrics sidecar (summarize_ci_run.py --summary-name <model>_<config>); the legs run in
separate jobs on separate clusters, so this runs at run level over the collected sidecars rather than
inside either leg. Pairing is by model: <model>_sc1 vs <model>_sc4.

This is the only place a leg's perf numbers surface, so a model with just one config present still gets
its table -- only the gain column is withheld. The gain is a pipeline-scaling number solely when both
sides measured the same request, so it is also withheld when chunk_size or num_chunks disagree. Stdlib
only -- this runs on a plain ubuntu runner with no tt-metal env.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# (json key, display name, unit, lower-is-better). Gain is always "how many times better sc4 is", so it
# inverts for throughput; that keeps every column readable as ">1 means the pipeline bought us something".
_FAMILIES = (
    ("chunk_time_ms", "chunk_time", "ms", True),
    ("ttft_s", "ttft", "s", True),
    ("throughput_tok_s", "throughput", "tok/s", False),
)
_BASE, _REF = "sc1", "sc4"


def _load(root):
    """model -> config -> record, over every *.json under root (download-artifact nests per artifact)."""
    by_model = {}
    for path in sorted(Path(root).rglob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"::warning::unreadable metrics sidecar {path}: {exc}")
            continue
        model, config = rec.get("model"), rec.get("config")
        if not model or not config:
            continue
        by_model.setdefault(model, {})[config] = rec
    return by_model


def _rows(base, ref, gain=True):
    rows = []
    for family, name, unit, lower_better in _FAMILIES:
        for label in sorted(set(base.get(family, {})) | set(ref.get(family, {}))):
            b, r = base.get(family, {}).get(label), ref.get(family, {}).get(label)
            metric = f"{name} {label}"
            if not gain or b is None or r is None or b <= 0 or r <= 0:
                cell = "-"
            else:
                cell = f"{(b / r) if lower_better else (r / b):.2f}x"
            fmt = lambda v: "-" if v is None else f"{v:,.3f} {unit}"  # noqa: E731
            rows.append([metric, fmt(b), fmt(r), cell])
    return rows


def _table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    render = lambda vals: "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"  # noqa: E731
    return [sep, render(headers), sep, *[render(r) for r in rows], sep]


def _comparable(base, ref):
    """Same request on both sides, or the ratio measures the request instead of the pipeline."""
    for key in ("chunk_size", "num_chunks"):
        if base.get(key) != ref.get(key):
            return f"{key} differs ({_BASE}={base.get(key)}, {_REF}={ref.get(key)})"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True, help="dir of *.json metrics sidecars (searched recursively)")
    args = ap.parse_args()

    lines = []
    by_model = _load(args.metrics_dir) if os.path.isdir(args.metrics_dir) else {}
    if not by_model:
        lines.append(f"No metrics sidecars found under {args.metrics_dir}; nothing to compare.")
    for model in sorted(by_model):
        configs = by_model[model]
        base, ref = configs.get(_BASE) or {}, configs.get(_REF) or {}
        lines.append(f"#### {model}")
        lines.append("")
        note = None
        if not base or not ref:
            note = f"only {', '.join(sorted(configs))} ran this run; no gain to compute"
        elif reason := _comparable(base, ref):
            note = f"{_BASE} and {_REF} measured different requests ({reason}); gain withheld"
        if note:
            lines.append(note)
            lines.append("")
        shape = ref or base
        lines.append("```text")
        lines.append(f"request: {shape['num_chunks']} chunks x {shape['chunk_size']} tok = {shape['max_seq']} tok")
        lines.append(f"gain = how many times better {_REF} is than {_BASE}")
        lines.append("note: chunk_time is one chunk's latency through the whole pipeline, which pipelining does not")
        lines.append("      shorten -- and a deeper pipeline pays more D2D hops, so <1x there is expected. The win")
        lines.append("      is in ttft/throughput.")
        lines += _table(["metric", _BASE, _REF, f"{_REF} gain"], _rows(base, ref, gain=not note))
        lines.append("```")
        lines.append("")

    block = "### disaggregated prefill scaling -- {} vs {}\n\n{}\n".format(_REF, _BASE, "\n".join(lines))
    print(block)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
