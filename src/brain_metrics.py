"""
Compatibility facade for the brain-network metric pipeline.

The implementation is split by processing order:

1. step01_eeg_io.py
2. step02_connectivity.py
3. step03_connection_strength.py
4. step04_graph_metrics.py
5. step05_hub_detection.py
6. step06_rich_club.py
7. step07_target_attack.py
8. step08_summary_metrics.py

Existing scripts can continue to use:

    import brain_metrics as bm
"""

from __future__ import annotations

from step01_eeg_io import read_eeg_set
from step02_connectivity import (
    _absolute_pearson_matrix,
    _absolute_profile_correlation,
    _signed_pearson_matrix,
    _topographical_profile,
    _validate_channel_time_data,
    _validate_connectivity_matrix,
    binarize,
    compute_ahofc,
    compute_associated_hofc,
    compute_envelope_correlation,
    compute_hofc,
    compute_lofc,
    compute_signed_lofc,
    compute_topographical_hofc,
    compute_topographical_profile_correlation,
    fisher_z_transform_connectivity,
    proportional_threshold,
)
from step03_connection_strength import average_connection_strength
from step04_graph_metrics import (
    _bfs_distance,
    clustering_coef_bin_dir,
    clustering_coef_bin_und,
    clustering_coef_wd,
    clustering_coef_wu,
    global_efficiency_weighted,
    graph_qc_metrics,
    global_efficiency_bin,
)
from step05_hub_detection import (
    betweenness_bin,
    betweenness_weighted,
    closeness_centrality,
    closeness_weighted,
    compute_ce_components,
    compute_ce,
    compute_weighted_ce,
    critic_objective_weights,
    degree_centrality,
    identify_traditional_hubs,
    identify_hub_nodes,
    identify_weighted_hub_nodes,
    strength_centrality,
    top_ratio_mask,
    topsis_closeness_scores,
)
from step06_rich_club import (
    _randmio_dir,
    _randmio_und,
    compute_binary_normalized_rich_club,
    compute_weighted_normalized_rich_club,
    rich_club_bd,
    rich_club_bu,
    weighted_rich_club_observed,
    weighted_rich_club,
)
from step07_target_attack import (
    compute_attack_by_order,
    compute_attack_curves,
    compute_centrality_ablation_attacks,
    compute_target_attack,
)
from step08_summary_metrics import compute_all_scalar_metrics

__all__ = [
    "read_eeg_set",
    "compute_envelope_correlation",
    "compute_lofc",
    "compute_signed_lofc",
    "fisher_z_transform_connectivity",
    "compute_topographical_profile_correlation",
    "compute_topographical_hofc",
    "compute_hofc",
    "compute_associated_hofc",
    "compute_ahofc",
    "binarize",
    "proportional_threshold",
    "global_efficiency_bin",
    "global_efficiency_weighted",
    "graph_qc_metrics",
    "degree_centrality",
    "betweenness_bin",
    "closeness_centrality",
    "strength_centrality",
    "betweenness_weighted",
    "closeness_weighted",
    "critic_objective_weights",
    "topsis_closeness_scores",
    "clustering_coef_bin_und",
    "clustering_coef_bin_dir",
    "clustering_coef_wu",
    "clustering_coef_wd",
    "compute_ce",
    "compute_ce_components",
    "compute_weighted_ce",
    "identify_hub_nodes",
    "identify_weighted_hub_nodes",
    "identify_traditional_hubs",
    "top_ratio_mask",
    "rich_club_bu",
    "rich_club_bd",
    "compute_binary_normalized_rich_club",
    "weighted_rich_club",
    "weighted_rich_club_observed",
    "compute_weighted_normalized_rich_club",
    "compute_target_attack",
    "compute_attack_by_order",
    "compute_attack_curves",
    "compute_centrality_ablation_attacks",
    "average_connection_strength",
    "compute_all_scalar_metrics",
]
