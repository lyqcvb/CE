"""
Step 16: CAUEEG analysis cohorts.

Matching uses only age and group of records that passed step15 (>= 50 clean
epochs, <= 3 bad channels).

Primary cohort: optimal 1:1:1 age matching. Triplet age types (a_HC, a_MCI,
a_AD) with pairwise |difference| <= 2 years are counted by an integer linear
program that maximizes the number of triplets subject to the age supply in
each group and to |mean age difference| <= 0.25 years for every group pair
(the greedy HC-anchored rule left AD ~1.6 years older). Ties are broken by the
smallest total within-triplet age difference. Subjects are then drawn at
random within age (seed 42); 100 further draws give the resampling cohorts.

Sensitivity cohort: every eligible record aged 60-85 years (age covariate).
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import caueeg_config as cfg

CALIPER = 2
MEAN_BALANCE = 0.25
N_RESAMPLES = 100
SENSITIVITY_AGES = (60, 85)


def solve_triplet_counts(
    supply: dict[str, pd.Series],
    caliper: int = CALIPER,
    balance: float = MEAN_BALANCE,
) -> dict[tuple[int, int, int], int]:
    """Integer program over age-triplet types; returns {(a_hc, a_mci, a_ad): count}."""
    ages = {g: supply[g][supply[g] > 0] for g in cfg.GROUPS}
    types = [
        (a, b, c)
        for a in ages["HC"].index
        for b in ages["MCI"].index
        for c in ages["AD"].index
        if max(abs(a - b), abs(a - c), abs(b - c)) <= caliper
    ]
    if not types:
        return {}
    t = np.array(types, dtype=float)
    n_var = len(types)

    rows, lb, ub = [], [], []
    for g_idx, g in enumerate(cfg.GROUPS):
        for age, count in ages[g].items():
            rows.append((t[:, g_idx] == age).astype(float))
            lb.append(0.0)
            ub.append(float(count))
    for i, j in itertools.combinations(range(3), 2):
        diff = t[:, j] - t[:, i]
        rows.append(diff - balance)
        lb.append(-np.inf)
        ub.append(0.0)
        rows.append(-diff - balance)
        lb.append(-np.inf)
        ub.append(0.0)

    spread = np.abs(t[:, 0] - t[:, 1]) + np.abs(t[:, 0] - t[:, 2]) + np.abs(t[:, 1] - t[:, 2])
    objective = -1.0 + 1e-3 * spread / (3 * caliper + 1)
    result = milp(
        c=objective,
        constraints=LinearConstraint(np.vstack(rows), lb, ub),
        integrality=np.ones(n_var),
        bounds=Bounds(0, np.inf),
    )
    if not result.success:
        raise RuntimeError(f"matching MILP failed: {result.message}")
    counts = np.round(result.x).astype(int)
    return {types[i]: int(counts[i]) for i in range(n_var) if counts[i] > 0}


def draw_cohort(eligible: pd.DataFrame, counts: dict[tuple[int, int, int], int], seed: int) -> pd.DataFrame:
    """Draw subjects at random within (group, age) to fill the matched triplet types."""
    rng = np.random.default_rng(seed)
    pools = {
        (g, age): list(rng.permutation(sub["serial"].to_numpy()))
        for (g, age), sub in eligible.groupby(["group", "age"])
    }
    rows = []
    triplet = 0
    for (a, b, c), count in sorted(counts.items()):
        for _ in range(count):
            triplet += 1
            for g, age in zip(cfg.GROUPS, (a, b, c)):
                rows.append({"triplet": triplet, "group": g, "age": age, "serial": pools[(g, age)].pop()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 1:1:1 年龄匹配与敏感性队列"))
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)

    pool = pd.read_csv(paths.results / "candidate_pool.csv", dtype={"serial": str})
    log = pd.read_csv(paths.results / "preproc_log.csv", dtype={"serial": str})
    eligible = pool.merge(log[["serial", "status"]], on="serial")
    eligible = eligible[eligible["status"] == "ok"].copy()

    supply = {g: eligible[eligible["group"] == g].groupby("age").size() for g in cfg.GROUPS}
    counts = solve_triplet_counts(supply)
    matched = draw_cohort(eligible, counts, cfg.SEED)
    matched = matched.merge(eligible[["serial", "symptom", "split"]], on="serial")
    matched.to_csv(paths.results / "cohort_matched_111.csv", index=False, encoding="utf-8-sig")

    resamples = pd.concat(
        [draw_cohort(eligible, counts, cfg.SEED + k).assign(draw=k) for k in range(1, N_RESAMPLES + 1)],
        ignore_index=True,
    )
    resamples.to_csv(paths.results / "cohort_resamples.csv", index=False, encoding="utf-8-sig")

    lo, hi = SENSITIVITY_AGES
    sensitivity = eligible[eligible["age"].between(lo, hi)][["serial", "group", "age", "symptom", "split"]]
    sensitivity.to_csv(paths.results / "cohort_60_85.csv", index=False, encoding="utf-8-sig")

    attrition = (
        pool.merge(log[["serial", "status"]], on="serial", how="left")
        .assign(status=lambda d: d["status"].fillna("not_preprocessed"))
        .groupby(["group", "status"]).size().unstack(fill_value=0)
    )
    attrition.to_csv(paths.results / "cohort_attrition.csv", encoding="utf-8-sig")

    summary = []
    for name, table in (("matched_111", matched), ("age_60_85", sensitivity), ("eligible_all", eligible)):
        for g in cfg.GROUPS:
            ages = table.loc[table["group"] == g, "age"]
            summary.append({"cohort": name, "group": g, "n": len(ages), "age_mean": ages.mean(),
                            "age_sd": ages.std(), "age_min": ages.min(), "age_max": ages.max()})
    summary = pd.DataFrame(summary)
    summary.to_csv(paths.results / "cohort_summary.csv", index=False, encoding="utf-8-sig")
    print(attrition.to_string())
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
