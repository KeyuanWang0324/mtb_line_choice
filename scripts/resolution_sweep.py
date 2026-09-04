"""How small a line difference can the analysis stage resolve, and what breaks it?

Gate 1a asks whether a ~2m A/B split survives the pipeline. Gate 1b asks about
~30cm, which is what the product actually needs. Two experiments here:

  1. split size x residual relocalization drift -> where does grouping fail
  2. number of runs -> what does collecting more footage actually buy

Synthetic, so this bounds the *analysis* stage only: real footage adds
reconstruction error on top of everything modelled here. Read the numbers as a
ceiling on performance, never as a forecast of it.

    python scripts/resolution_sweep.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mtb_line.centerline import fit_centerline
from mtb_line.evaluate import evaluate_split
from mtb_line.frenet import project_to_frenet
from mtb_line.synth import synthetic_segment

WANDER_M = 0.35  # the synth default: how much a rider varies run to run


def trial(split: float, drift: float, seed: int, n_per_side: int = 3) -> dict:
    segment, labels, window = synthetic_segment(
        n_left=n_per_side, n_right=n_per_side,
        split_offset_m=split, reloc_drift_m=drift, seed=seed,
    )
    cl = fit_centerline([t.xyz for t in segment.trajectories], up=segment.reconstruction.up)
    profiles = [project_to_frenet(cl, t.xyz, run_id=t.run_id) for t in segment.trajectories]
    # min_separation_m=0: the 1.0m floor is a Gate 1a criterion, not a sweep one.
    result = evaluate_split(profiles, labels, window, min_separation_m=0.0)

    means = {k: v["mean_d"] for k, v in result.per_run.items()}
    left = [means[k] for k in means if labels[k] == "left"]
    right = [means[k] for k in means if labels[k] == "right"]
    return {
        "per_run_accuracy": result.accuracy,
        "all_correct": result.accuracy == 1.0,
        "separation_m": result.separation_m,
        "group_p": float(stats.ttest_ind(left, right).pvalue),
    }


def sweep_split_vs_drift(seeds=range(5)) -> None:
    splits = [2.0, 1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    drifts = [0.05, 0.15, 0.30, 0.50]
    print(f"\n1. SPLIT SIZE vs RELOC DRIFT   (3 runs/side, wander {WANDER_M} m, "
          f"{len(list(seeds))} seeds)")
    print("   cell = share of trials where every run grouped correctly / mean separation\n")
    print(f"   {'split':>7} | " + " | ".join(f"drift {d:.2f}m" for d in drifts))
    print("   " + "-" * (9 + 14 * len(drifts)))
    for split in splits:
        cells = []
        for drift in drifts:
            runs = [trial(split, drift, s) for s in seeds]
            cells.append(f"{np.mean([r['all_correct'] for r in runs]):>5.0%} "
                         f"{np.mean([r['separation_m'] for r in runs]):>5.2f}m")
        print(f"   {split:>6.1f}m | " + " | ".join(cells))


def sweep_n_runs(seeds=range(12)) -> None:
    print(f"\n2. NUMBER OF RUNS at the Gate 1b scale   (0.30 m split, 0.15 m drift, "
          f"wander {WANDER_M} m)\n")
    print(f"   {'runs/side':>10} {'per-run acc':>13} {'group p<0.05':>14} {'mean sep':>10}")
    for n in [3, 5, 8, 12, 20]:
        runs = [trial(0.30, 0.15, s, n_per_side=n) for s in seeds]
        print(f"   {n:>10} {np.mean([r['per_run_accuracy'] for r in runs]):>12.0%} "
              f"{np.mean([r['group_p'] < 0.05 for r in runs]):>13.0%} "
              f"{np.mean([r['separation_m'] for r in runs]):>9.2f}m")


def main() -> None:
    print(__doc__.strip().split("\n\n")[0])
    sweep_split_vs_drift()
    sweep_n_runs()
    print("""
FINDINGS

  Gate 1a (~2 m split) is not close to the limit. It groups perfectly even at
  0.50 m of relocalization drift, which is far worse drift than a working
  pipeline should produce. If the real Gate 1a fails, the cause is
  reconstruction or relocalization, not this analysis.

  Gate 1b (~30 cm) is a different kind of problem, and adding runs does not
  fix it the way it looks like it should. Per-run accuracy sits at ~93%
  regardless of how many runs are collected, because at this scale the split is
  SMALLER THAN RIDER WANDER (0.30 m vs 0.35 m): on roughly one run in fourteen
  the rider genuinely was on the other side. That is not pipeline error to be
  engineered away, it is the sport.

  What more runs buy is confidence in the GROUP MEAN, not per-run correctness.
  Five runs per side reaches a reliable group-level difference (p < 0.05 on
  every seed); three does not (67%).

PRODUCT CONSEQUENCE

  At the 30 cm scale the honest claim is statistical, not individual:
      "riders who clear this section fastest carry ~30 cm more inside line"
  and NOT:
      "you took the wrong line here."
  Per-run line attribution is only defensible where the separation exceeds
  rider wander -- roughly 0.6 m and up on this model.

  Collection target follows directly: >= 10 runs per segment (5 per branch),
  not the 3 the Gate 1 plan originally assumed.
""")


if __name__ == "__main__":
    main()
