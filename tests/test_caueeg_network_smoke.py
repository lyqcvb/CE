from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from network_nulls import degree_preserving_rewire, strength_preserving_null, weight_permutation_null  # noqa: E402
from step02_connectivity import compute_signed_hofc, compute_topographical_hofc  # noqa: E402
from step04_graph_metrics import global_efficiency_bin, global_efficiency_weighted, largest_component_size  # noqa: E402
from step07_target_attack import compute_attack_by_order  # noqa: E402
from step16_caueeg_cohort import draw_cohort, solve_triplet_counts  # noqa: E402
from step19_caueeg_network import attack_curves_both, build_networks, density_graph  # noqa: E402


def _random_weight(n=19, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random((n, n))
    w = np.triu(w, 1)
    return w + w.T


def test_signed_hofc_keeps_sign_and_matches_abs_definition():
    rng = np.random.default_rng(2)
    signed = np.corrcoef(rng.standard_normal((8, 50)))
    shofc = compute_signed_hofc(signed)
    abs_from_signed = compute_topographical_hofc(signed, signed=False)
    assert (shofc < 0).any()
    assert np.allclose(np.abs(shofc), abs_from_signed)
    assert np.allclose(shofc, shofc.T)


def test_fast_paths_match_networkx():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = (rng.random((15, 15)) < 0.25).astype(float)
        a = np.triu(a, 1)
        a = a + a.T
        g = nx.from_numpy_array(a)
        assert np.isclose(global_efficiency_bin(a), nx.global_efficiency(g))
        assert largest_component_size(a) == max(len(c) for c in nx.connected_components(g))
    w = _random_weight(10, 4)
    gw = nx.from_numpy_array(w)
    for u, v, data in gw.edges(data=True):
        data["length"] = 1.0 / data["weight"]
    lengths = dict(nx.all_pairs_dijkstra_path_length(gw, weight="length"))
    expected = np.mean([1.0 / lengths[i][j] for i in range(10) for j in range(10) if i != j])
    assert np.isclose(global_efficiency_weighted(w), expected)


def test_null_models_preserve_their_invariants():
    rng = np.random.default_rng(5)
    w = _random_weight(19, 6)
    upper = np.triu_indices(19, 1)
    for null in (strength_preserving_null(w, rng), weight_permutation_null(w, rng)):
        assert np.allclose(np.sort(null[upper]), np.sort(w[upper]))
        assert np.allclose(null, null.T)
    sp = strength_preserving_null(w, rng)
    perm = weight_permutation_null(w, rng)
    err_sp = np.abs(sp.sum(1) - w.sum(1)).mean()
    err_perm = np.abs(perm.sum(1) - w.sum(1)).mean()
    assert err_sp < err_perm

    b = density_graph(w, 0.3)
    rewired = degree_preserving_rewire(b, rng)
    assert np.array_equal(rewired.sum(1), b.sum(1))
    assert not np.array_equal(rewired, b)


def test_fused_attack_curves_match_step07():
    rng = np.random.default_rng(7)
    b = density_graph(_random_weight(19, 8), 0.25)
    order = rng.permutation(19)
    both = attack_curves_both(b, order)
    assert np.allclose(both["eff"], compute_attack_by_order(b, order))
    assert np.allclose(both["lcc"], compute_attack_by_order(b, order, outcome="lcc"))


def test_density_graph_never_keeps_zero_weight_edges():
    w = _random_weight(19, 9)
    w[w < 0.8] = 0.0
    b = density_graph(w, 0.40)
    assert np.all(w[b > 0] > 0)


def test_build_networks_shapes():
    signed = np.corrcoef(np.random.default_rng(10).standard_normal((19, 200)))
    np.fill_diagonal(signed, 0.0)
    nets = build_networks(signed, "aecc")
    for fc in ("LOFC", "HOFC", "sHOFC"):
        graph, raw = nets[fc]
        assert graph.shape == (19, 19)
        assert (graph >= 0).all()
        assert np.allclose(np.diag(graph), 0)
    assert (nets["sHOFC"][1] < 0).any()


def test_matching_balances_ages_and_is_deterministic():
    rng = np.random.default_rng(11)
    rows = []
    for g, (mu, n) in {"HC": (64, 120), "MCI": (72, 150), "AD": (75, 120)}.items():
        ages = np.clip(np.round(rng.normal(mu, 7, n)), 55, 90).astype(int)
        rows += [{"serial": f"{g}{i:03d}", "group": g, "age": a} for i, a in enumerate(ages)]
    eligible = pd.DataFrame(rows)
    supply = {g: eligible[eligible["group"] == g].groupby("age").size() for g in ("HC", "MCI", "AD")}
    counts = solve_triplet_counts(supply)
    cohort = draw_cohort(eligible, counts, seed=42)
    assert cohort.equals(draw_cohort(eligible, counts, seed=42))
    means = cohort.groupby("group")["age"].mean()
    assert means.max() - means.min() <= 0.25 + 1e-9
    wide = cohort.pivot(index="triplet", columns="group", values="age")
    assert (wide.max(axis=1) - wide.min(axis=1)).max() <= 2
    assert cohort["serial"].is_unique
