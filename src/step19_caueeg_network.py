"""
Step 19: Network metrics for CAUEEG connectivity matrices.

For every record x measure (aec/aecc/wpli) x band, three networks are built:
- LOFC  : non-negative connectivity graph from step18;
- HOFC  : legacy |corr| of Fisher-z LOFC profiles (abs at both levels);
- sHOFC : signed HOFC from the signed LOFC; topology uses its positive part,
          and the signed mean / negative-edge share are reported.

Metrics per network (column prefixes):
- MeanConn, Eglob_w, Clust_w                        weighted, threshold-free
- Eglob_w_norm, Clust_w_norm                        / strength-preserving nulls
- Eglob_w_permnorm, Clust_w_permnorm                / weight-permutation nulls
- Eglob_bin_auc, Clust_bin_auc (+ _norm)            fixed density 0.20-0.40
- Thr035_Density, Thr035_Eglob, Thr035_Clust        legacy absolute threshold
- AttackAUC_{order}_{eff|lcc}                       order in ce/degree/betweenness/
                                                    closeness/topsis/random,
                                                    averaged over densities
- AttackAUCdiff_{order}_{outcome}                   minus the random-order AUC
- AttackAUCcv_{ce|betweenness}_{outcome}            ranks from one half of the
                                                    epochs, attack the other half
- HubJaccard_CE_BC                                  top-4 hub overlap
- Stability_r                                       mean pairwise r over 5 chunks
Nulls, random-order attacks and cross-validated attacks are computed for the
primary measure (AEC-c) only.

Output: output/caueeg/network_metrics.csv and attack_curves.csv.
"""

from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import caueeg_config as cfg
from network_nulls import degree_preserving_rewire, strength_preserving_null, weight_permutation_null
from step02_connectivity import binarize, compute_signed_hofc, compute_topographical_hofc, proportional_threshold
from step04_graph_metrics import _bfs_distance, clustering_coef_bin_und, clustering_coef_wu, global_efficiency_bin, global_efficiency_weighted
from step05_hub_detection import compute_ce_components, top_ratio_mask
from step07_target_attack import _attack_order, _metric_order

FC_TYPES = ["LOFC", "HOFC", "sHOFC"]
ORDERS = ["ce", "degree", "betweenness", "closeness", "topsis"]
OUTCOMES = {"eff": "efficiency", "lcc": "lcc"}
BINARY_NULL_REPEATS = 20
CV_RANDOM_REPEATS = 50


def _graph_from_signed(signed: np.ndarray, measure: str) -> np.ndarray:
    graph = np.abs(signed) if measure == "aec" else np.clip(signed, 0.0, None)
    graph = 0.5 * (graph + graph.T)
    np.fill_diagonal(graph, 0.0)
    return graph


def build_networks(signed: np.ndarray, measure: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {fc: (non-negative graph, signed matrix)}; signed equals graph except for sHOFC."""
    lofc = _graph_from_signed(signed, measure)
    with_diag = lofc.copy()
    np.fill_diagonal(with_diag, 1.0)
    hofc = compute_topographical_hofc(with_diag)
    signed_diag = 0.5 * (signed + signed.T)
    np.fill_diagonal(signed_diag, 1.0)
    shofc = compute_signed_hofc(signed_diag)
    shofc_pos = np.clip(shofc, 0.0, None)
    np.fill_diagonal(shofc_pos, 0.0)
    return {"LOFC": (lofc, lofc), "HOFC": (hofc, hofc), "sHOFC": (shofc_pos, shofc)}


def density_graph(weight: np.ndarray, density: float) -> np.ndarray:
    """Fixed-density binary graph; zero-weight edges are never kept (sparse AEC-c/sHOFC)."""
    return proportional_threshold(weight, density) * (weight > 0.0)


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _clust_w(weight: np.ndarray) -> float:
    return float(np.mean(clustering_coef_wu(weight)))


def _clust_bin(graph: np.ndarray) -> float:
    return float(np.mean(clustering_coef_bin_und(graph)))


def _null_ratio(observed: float, null_values: list[float]) -> float:
    mean = float(np.mean(null_values))
    return observed / mean if mean > 0 else np.nan


def _auc(values: list[float], x: list[float]) -> float:
    return float(np.trapezoid(values, x) / (x[-1] - x[0]))


def _orders(graph: np.ndarray) -> dict[str, np.ndarray]:
    comp = compute_ce_components(graph)
    return {
        "ce": _attack_order(comp["CE"], comp["S"], graph, directed=False),
        "degree": _metric_order(comp["Degree"]),
        "betweenness": _metric_order(comp["Betweenness"]),
        "closeness": _metric_order(comp["Closeness"]),
        "topsis": _metric_order(comp["S"]),
    }


def attack_curves_both(graph: np.ndarray, order: np.ndarray) -> dict[str, np.ndarray]:
    """
    Efficiency and LCC curves (baseline 1 at step 0) from one BFS per removal step.

    Same semantics as step07.compute_attack_by_order(outcome="efficiency"/"lcc").
    """
    working = (np.asarray(graph) > 0).astype(np.float64)
    working = np.maximum(working, working.T)
    np.fill_diagonal(working, 0.0)
    n = working.shape[0]

    def measure(adj):
        dist = _bfs_distance(adj)
        finite = np.isfinite(dist)
        inv = np.zeros_like(dist)
        off = finite & (dist > 0)
        inv[off] = 1.0 / dist[off]
        return inv.sum() / (n * (n - 1)), finite.sum(axis=1).max()

    eff0, lcc0 = measure(working)
    eff = np.zeros(n + 1)
    lcc = np.zeros(n + 1)
    if eff0 > 0:
        eff[0] = 1.0
    lcc[0] = 1.0
    for step, node in enumerate(order[:n], start=1):
        working[node, :] = 0.0
        working[:, node] = 0.0
        e, c = measure(working)
        eff[step] = e / eff0 if eff0 > 0 else 0.0
        lcc[step] = c / lcc0
    return {"eff": eff, "lcc": lcc}


def _auc_curve(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve) / (curve.size - 1))


def _random_aucs(graph: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    n = graph.shape[0]
    curves = [attack_curves_both(graph, rng.permutation(n)) for _ in range(repeats)]
    mean_eff = np.mean([c["eff"] for c in curves], axis=0)
    mean_lcc = np.mean([c["lcc"] for c in curves], axis=0)
    return {"eff": _auc_curve(mean_eff), "lcc": _auc_curve(mean_lcc)}, mean_eff


def weighted_metrics(graph: np.ndarray, signed: np.ndarray, full: bool, rng: np.random.Generator) -> dict:
    out = {
        "MeanConn": float(_upper(graph).mean()),
        "Eglob_w": global_efficiency_weighted(graph),
        "Clust_w": _clust_w(graph),
    }
    if signed is not graph:
        values = _upper(signed)
        out["MeanConn_signed"] = float(values.mean())
        out["NegEdgeFraction"] = float((values < 0).mean())
        out["NegMean"] = float(values[values < 0].mean()) if (values < 0).any() else 0.0
    if full:
        for suffix, null_fn in (("norm", strength_preserving_null), ("permnorm", weight_permutation_null)):
            nulls = [null_fn(graph, rng) for _ in range(cfg.NULL_REPEATS)]
            out[f"Eglob_w_{suffix}"] = _null_ratio(out["Eglob_w"], [global_efficiency_weighted(n) for n in nulls])
            out[f"Clust_w_{suffix}"] = _null_ratio(out["Clust_w"], [_clust_w(n) for n in nulls])
    return out


def density_metrics(
    graph: np.ndarray,
    halves: tuple[np.ndarray, np.ndarray] | None,
    full: bool,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    densities = cfg.DENSITIES
    per_density: dict[str, list[float]] = {}
    curves: dict[str, list[np.ndarray]] = {}

    def add(key, value):
        per_density.setdefault(key, []).append(value)

    for density in densities:
        binary = density_graph(graph, density)
        eglob, clust = global_efficiency_bin(binary), _clust_bin(binary)
        add("Eglob_bin", eglob)
        add("Clust_bin", clust)
        if full:
            nulls = [degree_preserving_rewire(binary, rng) for _ in range(BINARY_NULL_REPEATS)]
            add("Eglob_bin_norm", _null_ratio(eglob, [global_efficiency_bin(n) for n in nulls]))
            add("Clust_bin_norm", _null_ratio(clust, [_clust_bin(n) for n in nulls]))

        orders = _orders(binary)
        ce_hubs = set(np.flatnonzero(top_ratio_mask(compute_ce_components(binary)["CE"], 0.20)))
        bc_hubs = set(np.flatnonzero(top_ratio_mask(compute_ce_components(binary)["Betweenness"], 0.20)))
        add("HubJaccard_CE_BC", len(ce_hubs & bc_hubs) / max(len(ce_hubs | bc_hubs), 1))
        for name, order in orders.items():
            curves_both = attack_curves_both(binary, order)
            for tag in OUTCOMES:
                add(f"AttackAUC_{name}_{tag}", _auc_curve(curves_both[tag]))
            if name in ("ce", "betweenness"):
                curves.setdefault(name, []).append(curves_both["eff"])
        if full:
            rand, mean_eff = _random_aucs(binary, cfg.RANDOM_ATTACK_REPEATS, rng)
            for tag in OUTCOMES:
                add(f"AttackAUC_random_{tag}", rand[tag])
            curves.setdefault("random", []).append(mean_eff)

        if full and halves is not None:
            bin_a = density_graph(halves[0], density)
            bin_b = density_graph(halves[1], density)
            orders_a, orders_b = _orders(bin_a), _orders(bin_b)
            rand_a, _ = _random_aucs(bin_a, CV_RANDOM_REPEATS, rng)
            rand_b, _ = _random_aucs(bin_b, CV_RANDOM_REPEATS, rng)
            for name in ("ce", "betweenness"):
                ab = attack_curves_both(bin_b, orders_a[name])
                ba = attack_curves_both(bin_a, orders_b[name])
                for tag in OUTCOMES:
                    add(f"AttackAUCcv_{name}_{tag}",
                        0.5 * ((_auc_curve(ab[tag]) - rand_b[tag]) + (_auc_curve(ba[tag]) - rand_a[tag])))

    out = {}
    for key, values in per_density.items():
        if key in ("Eglob_bin", "Clust_bin", "Eglob_bin_norm", "Clust_bin_norm"):
            out[f"{key}_auc"] = _auc(values, densities)
        else:
            out[key] = float(np.mean(values))
    if full:
        for tag in OUTCOMES:
            for name in ORDERS:
                out[f"AttackAUCdiff_{name}_{tag}"] = out[f"AttackAUC_{name}_{tag}"] - out[f"AttackAUC_random_{tag}"]
    return out, {k: np.mean(v, axis=0) for k, v in curves.items()}


def stability(chunks: list[np.ndarray]) -> float:
    pairs = [np.corrcoef(_upper(a), _upper(b))[0, 1] for a, b in itertools.combinations(chunks, 2)]
    return float(np.nanmean(pairs))


def network_record(serial: str, project_root, measures: list[str]) -> tuple[list[dict], list[dict]]:
    paths = cfg.CaueegPaths(project_root)
    rng = np.random.default_rng(cfg.SEED + int(serial))
    rows, curve_rows = [], []
    for measure in measures:
        full = measure == cfg.PRIMARY_MEASURE
        for band in cfg.BANDS:
            bundle = np.load(paths.fc / measure / band / f"{serial}.npz")
            networks = build_networks(bundle["full"], measure)
            chunk_nets = [build_networks(m, measure) for m in bundle["chunks"]]
            half_nets = [build_networks(m, measure) for m in bundle["halves"]]
            for fc in FC_TYPES:
                graph, signed = networks[fc]
                row = {"serial": serial, "measure": measure, "band": band, "fc": fc}
                row.update(weighted_metrics(graph, signed, full, rng))
                halves = (half_nets[0][fc][0], half_nets[1][fc][0])
                dens, curves = density_metrics(graph, halves, full, rng)
                row.update(dens)
                thr = binarize(graph, cfg.THRESHOLD_SENSITIVITY)
                row["Thr035_Density"] = float(_upper(thr).mean())
                row["Thr035_Eglob"] = global_efficiency_bin(thr)
                row["Thr035_Clust"] = _clust_bin(thr)
                row["Stability_r"] = stability([net[fc][1] for net in chunk_nets])
                rows.append(row)
                if full:
                    for order, curve in curves.items():
                        for step, value in enumerate(curve):
                            curve_rows.append({"serial": serial, "band": band, "fc": fc,
                                               "order": order, "step": step, "value": float(value)})
    return rows, curve_rows


def _job(args):
    return network_record(*args)


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 网络指标"))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--measures", nargs="*", default=cfg.MEASURES)
    parser.add_argument("--serials", nargs="*")
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)
    if args.serials:
        serials = args.serials
    else:
        log = pd.read_csv(paths.results / "preproc_log.csv", dtype={"serial": str})
        serials = log.loc[log["status"] == "ok", "serial"].tolist()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_job, [(s, args.project_root, args.measures) for s in serials], chunksize=2))
    paths.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rows, _ in results for r in rows]).to_csv(
        paths.output / "network_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([r for _, rows in results for r in rows]).to_csv(
        paths.output / "attack_curves.csv", index=False, encoding="utf-8-sig")
    print(f"network metrics for {len(serials)} records")


if __name__ == "__main__":
    main()
