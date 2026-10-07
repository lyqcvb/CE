"""
Step 20: Pre-specified statistics for the CAUEEG replication
(docs/caueeg_analysis_plan.md).

Primary: 8 AEC-c endpoints, each z-scored within band and fitted with
MixedLM value ~ group + band + age + (1|subject); group Wald chi2 (df=2) and
group x band interaction (separate model) are BH-FDR corrected across the 8
endpoints.

Outputs (results/caueeg/):
stats_primary.csv, stats_bandwise.csv, stats_power_adjusted.csv,
stats_power_adjusted_bandwise.csv (per band, all three measures),
stats_robustness.csv, stats_resampling.csv, stats_resampling_summary.csv,
stats_ce_vs_betweenness.csv, stats_order_interaction.csv, stats_qc.csv,
stats_spectral.csv, stats_nonparametric_bandwise.csv,
hofc_convergence_summary.csv, group_descriptives.csv
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import caueeg_config as cfg
from step11_statistical_analysis import _cliffs_delta, _fdr_bh

warnings.filterwarnings("ignore")

PRIMARY_ENDPOINTS = [
    ("LOFC", "MeanConn"),
    ("LOFC", "Eglob_w_norm"),
    ("LOFC", "Clust_w_norm"),
    ("LOFC", "AttackAUCdiff_ce_eff"),
    ("sHOFC", "MeanConn_signed"),
    ("sHOFC", "Eglob_w_norm"),
    ("sHOFC", "Clust_w_norm"),
    ("sHOFC", "AttackAUCdiff_ce_eff"),
]
GROUP_TERM = "C(group, Treatment('HC'))"
CONTRASTS = [("MCI", "HC"), ("AD", "HC"), ("AD", "MCI")]


# --------------------------------------------------------------------------- models
def _prepare(df: pd.DataFrame, value: str, covariates: list[str], by_band: bool = True) -> pd.DataFrame:
    cols = ["serial", "group", "age", value] + [c for c in covariates if c not in ("age_z", "age_z2")]
    if by_band:
        cols.append("band")
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if by_band:
        d["y"] = d.groupby("band")[value].transform(lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0)
    else:
        sd = d[value].std(ddof=1)
        d["y"] = (d[value] - d[value].mean()) / sd if sd > 0 else 0.0
    d["age_z"] = (d["age"] - d["age"].mean()) / d["age"].std(ddof=1)
    d["age_z2"] = d["age_z"] ** 2
    for c in covariates:
        if c in ("age_z", "age_z2"):
            continue
        sd = d[c].std(ddof=1)
        d[c] = (d[c] - d[c].mean()) / sd if sd > 0 else 0.0
    return d


def _fit_mixed(formula: str, d: pd.DataFrame):
    """
    REML MixedLM; keeps the converged fit with the highest log-likelihood.

    lbfgs alone can stop at the start values (all fixed effects 0) when the
    random-intercept variance is on the boundary, so powell is always tried
    too; nm/cg are fallbacks. Returns None if every optimizer fails.
    """
    best = None
    for method in ("lbfgs", "powell", "nm", "cg"):
        if best is not None and method in ("nm", "cg"):
            break
        try:
            res = smf.mixedlm(formula, d, groups=d["serial"]).fit(reml=True, method=method)
        except (np.linalg.LinAlgError, ValueError):
            continue
        ok = res.converged and np.all(np.isfinite(res.fe_params)) and np.all(
            np.isfinite(np.diag(res.cov_params().to_numpy())))
        if ok and (best is None or res.llf > best.llf + 1e-9):
            best = res
    return best


def _usable_covariates(d: pd.DataFrame, covariates: list[str]) -> list[str]:
    return [c for c in covariates if d[c].std(ddof=1) > 0]


def _group_tests(params: pd.Series, cov: pd.DataFrame) -> dict:
    names = {g: f"{GROUP_TERM}[T.{g}]" for g in ("MCI", "AD")}
    idx = [names["MCI"], names["AD"]]
    b = params[idx].to_numpy()
    v = cov.loc[idx, idx].to_numpy()
    chi2 = float(b @ np.linalg.solve(v, b))
    out = {"group_chi2": chi2, "group_p": float(stats.chi2.sf(chi2, 2))}
    for hi, lo in CONTRASTS:
        vec = np.zeros(2)
        if hi != "HC":
            vec[idx.index(names[hi])] += 1
        if lo != "HC":
            vec[idx.index(names[lo])] -= 1
        est = float(vec @ b)
        se = float(np.sqrt(vec @ v @ vec))
        z = est / se if se > 0 else np.nan
        key = f"{hi}_vs_{lo}"
        out[f"{key}_est"] = est
        out[f"{key}_se"] = se
        out[f"{key}_ci_lo"] = est - 1.96 * se
        out[f"{key}_ci_hi"] = est + 1.96 * se
        out[f"{key}_p"] = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return out


def fit_lmm(df: pd.DataFrame, value: str, covariates: list[str] | None = None, interaction: bool = True) -> dict:
    """z-scored-within-band MixedLM: group main effect, contrasts, and group x band interaction."""
    covariates = list(covariates or ["age_z"])
    d = _prepare(df, value, covariates)
    if d["serial"].nunique() < 10 or d["group"].nunique() < 3:
        return {"n_subjects": d["serial"].nunique()}
    covariates = _usable_covariates(d, covariates)
    rhs = " + ".join([GROUP_TERM, "C(band)"] + covariates)
    res = _fit_mixed(f"y ~ {rhs}", d)
    out = {"n_subjects": int(d["serial"].nunique()), "n_obs": int(len(d)), "covariates": ";".join(covariates)}
    if res is None:
        out["fit_error"] = "mixedlm failed"
        return out
    out["converged"] = bool(res.converged)
    out.update(_group_tests(res.fe_params, res.cov_params()))
    for g in cfg.GROUPS:
        out[f"n_{g}"] = int(d.loc[d["group"] == g, "serial"].nunique())
    if interaction:
        rhs_int = " + ".join([f"{GROUP_TERM} * C(band)"] + covariates)
        res_int = _fit_mixed(f"y ~ {rhs_int}", d)
        if res_int is None:
            return out
        terms = [t for t in res_int.fe_params.index if ":" in t]
        b = res_int.fe_params[terms].to_numpy()
        v = res_int.cov_params().loc[terms, terms].to_numpy()
        chi2 = float(b @ np.linalg.pinv(v) @ b)
        out["interaction_chi2"] = chi2
        out["interaction_df"] = len(terms)
        out["interaction_p"] = float(stats.chi2.sf(chi2, len(terms)))
    return out


def fit_ols(d: pd.DataFrame, value: str, covariates: list[str]) -> dict:
    d = _prepare(d, value, covariates, by_band=False)
    if d["group"].nunique() < 3:
        return {}
    covariates = _usable_covariates(d, covariates) or ["1"]
    res = smf.ols(f"y ~ {GROUP_TERM} + " + " + ".join(covariates), d).fit()
    out = {"n": int(len(d))}
    out.update(_group_tests(res.params, res.cov_params()))
    return out


def _fdr(df: pd.DataFrame, col: str, out_col: str, by: str | None = None) -> pd.DataFrame:
    if df.empty or col not in df:
        return df
    if by is None:
        df[out_col] = _fdr_bh(df[col].to_numpy(dtype=float))
    else:
        df[out_col] = np.nan
        for _, idx in df.groupby(by).groups.items():
            df.loc[idx, out_col] = _fdr_bh(df.loc[idx, col].to_numpy(dtype=float))
    return df


# --------------------------------------------------------------------------- data
def load_tables(paths: cfg.CaueegPaths) -> dict[str, pd.DataFrame]:
    read = lambda name: pd.read_csv(paths.results / name, dtype={"serial": str})  # noqa: E731
    tables = {
        "matched": read("cohort_matched_111.csv"),
        "sens": read("cohort_60_85.csv"),
        "resamples": read("cohort_resamples.csv"),
        "log": read("preproc_log.csv"),
        "spectral": read("spectral_global.csv"),
        "net": pd.read_csv(paths.output / "network_metrics.csv", dtype={"serial": str}),
    }
    conv = paths.results / "hofc_convergence_subject.csv"
    tables["convergence"] = pd.read_csv(conv, dtype={"serial": str}) if conv.exists() else pd.DataFrame()
    power = tables["spectral"].melt(id_vars="serial", value_vars=[f"logabs_{b}" for b in cfg.BANDS],
                                    var_name="band", value_name="logpow")
    power["band"] = power["band"].str.replace("logabs_", "", regex=False)
    tables["power"] = power
    return tables


def network_long(t: dict, cohort: pd.DataFrame, measure: str, fc: str) -> pd.DataFrame:
    net = t["net"][(t["net"]["measure"] == measure) & (t["net"]["fc"] == fc)]
    d = cohort[["serial", "group", "age"]].merge(net, on="serial")
    d = d.merge(t["power"], on=["serial", "band"], how="left")
    d = d.merge(t["spectral"][["serial", "ap_offset", "ap_exponent"]], on="serial", how="left")
    return d.merge(t["log"][["serial", "n_ica_excluded", "n_bad_channels"]], on="serial", how="left")


def _metric_for(measure: str, metric: str, available: pd.Index) -> str:
    if metric in available:
        return metric
    # AttackAUCdiff needs random-order attacks, computed for the primary measure only.
    return metric.replace("AttackAUCdiff", "AttackAUC") if metric.startswith("AttackAUCdiff") else metric


# --------------------------------------------------------------------------- analyses
def primary(t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, band_rows = [], []
    for fc, metric in PRIMARY_ENDPOINTS:
        d = network_long(t, t["matched"], cfg.PRIMARY_MEASURE, fc)
        rows.append({"measure": cfg.PRIMARY_MEASURE, "fc": fc, "metric": metric, **fit_lmm(d, metric)})
        for band in cfg.BANDS:
            band_rows.append({"fc": fc, "metric": metric, "band": band,
                              **fit_ols(d[d["band"] == band], metric, ["age_z"])})
    table = pd.DataFrame(rows)
    table = _fdr(table, "group_p", "group_p_fdr")
    table = _fdr(table, "interaction_p", "interaction_p_fdr")
    bands = pd.DataFrame(band_rows)
    for hi, lo in CONTRASTS:
        bands = _fdr(bands, f"{hi}_vs_{lo}_p", f"{hi}_vs_{lo}_p_fdr", by=["fc", "metric"])
    bands = _fdr(bands, "group_p", "group_p_fdr", by=["fc", "metric"])
    return table, bands


def power_adjusted(t: dict, primary_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fc, metric in PRIMARY_ENDPOINTS:
        d = network_long(t, t["matched"], cfg.PRIMARY_MEASURE, fc)
        base = primary_table[(primary_table["fc"] == fc) & (primary_table["metric"] == metric)].iloc[0]
        for label, covs in (("band_logpower", ["age_z", "logpow"]), ("aperiodic", ["age_z", "ap_offset", "ap_exponent"])):
            res = fit_lmm(d, metric, covs, interaction=False)
            row = {"fc": fc, "metric": metric, "adjustment": label, **res}
            for hi, lo in CONTRASTS:
                key = f"{hi}_vs_{lo}_est"
                row[f"{hi}_vs_{lo}_attenuation"] = 1 - abs(res.get(key, np.nan)) / abs(base[key]) if base[key] else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    return _fdr(table, "group_p", "group_p_fdr", by="adjustment")


def power_adjusted_bandwise(t: dict) -> pd.DataFrame:
    """Per-band OLS value ~ group + age (+ log band power or relative power); attenuation of contrasts."""
    rows = []
    rel = t["spectral"].melt(id_vars="serial", value_vars=[f"rel_{b}" for b in cfg.BANDS],
                             var_name="band", value_name="relpow")
    rel["band"] = rel["band"].str.replace("rel_", "", regex=False)
    for measure in cfg.MEASURES:
        for fc, metric in PRIMARY_ENDPOINTS:
            d = network_long(t, t["matched"], measure, fc).merge(rel, on=["serial", "band"], how="left")
            metric_used = _metric_for(measure, metric, d.columns[d.notna().any()])
            for band in cfg.BANDS:
                sub = d[d["band"] == band]
                base = fit_ols(sub, metric_used, ["age_z"])
                for label, covs in (("none", ["age_z"]), ("log_power", ["age_z", "logpow"]),
                                    ("relative_power", ["age_z", "relpow"])):
                    res = base if label == "none" else fit_ols(sub, metric_used, covs)
                    row = {"measure": measure, "fc": fc, "metric": metric_used, "band": band, "adjustment": label, **res}
                    for hi, lo in CONTRASTS:
                        key = f"{hi}_vs_{lo}_est"
                        row[f"{hi}_vs_{lo}_attenuation"] = (1 - abs(res[key]) / abs(base[key])) if base.get(key) else np.nan
                    rows.append(row)
    table = pd.DataFrame(rows)
    return _fdr(table, "group_p", "group_p_fdr", by=["measure", "adjustment"])


def robustness(t: dict) -> pd.DataFrame:
    rows = []

    def run(family, cohort, measure, fc, metric, covs=None):
        d = network_long(t, cohort, measure, fc)
        metric_used = _metric_for(measure, metric, d.columns[d.notna().any()])
        if metric_used not in d or d[metric_used].isna().all():
            return
        rows.append({"family": family, "measure": measure, "fc": fc, "metric": metric_used,
                     **fit_lmm(d, metric_used, covs, interaction=False)})

    for measure in ("aec", "wpli"):
        for fc, metric in PRIMARY_ENDPOINTS:
            run(f"measure_{measure}", t["matched"], measure, fc, metric)
    for metric in ("MeanConn", "Eglob_w_norm", "Clust_w_norm", "AttackAUCdiff_ce_eff"):
        run("legacy_abs_HOFC", t["matched"], cfg.PRIMARY_MEASURE, "HOFC", metric)
    for fc in ("LOFC", "HOFC", "sHOFC"):
        for metric in ("Eglob_w", "Clust_w", "Eglob_w_permnorm", "Clust_w_permnorm"):
            run("weighted_raw_and_permnorm", t["matched"], cfg.PRIMARY_MEASURE, fc, metric)
        for metric in ("Eglob_bin_auc", "Clust_bin_auc", "Eglob_bin_norm_auc", "Clust_bin_norm_auc"):
            run("fixed_density", t["matched"], cfg.PRIMARY_MEASURE, fc, metric)
        for measure in ("aec", cfg.PRIMARY_MEASURE):
            for metric in ("Thr035_Density", "Thr035_Eglob", "Thr035_Clust"):
                run(f"threshold_0.35_{measure}", t["matched"], measure, fc, metric)
        for order in ("ce", "degree", "betweenness", "closeness", "topsis"):
            for outcome in ("eff", "lcc"):
                run("attack_orders", t["matched"], cfg.PRIMARY_MEASURE, fc, f"AttackAUCdiff_{order}_{outcome}")
        for name in ("ce", "betweenness"):
            for outcome in ("eff", "lcc"):
                run("attack_crossvalidated", t["matched"], cfg.PRIMARY_MEASURE, fc, f"AttackAUCcv_{name}_{outcome}")
    for fc, metric in PRIMARY_ENDPOINTS:
        run("qc_covariates", t["matched"], cfg.PRIMARY_MEASURE, fc, metric,
            ["age_z", "n_ica_excluded", "n_bad_channels", "Stability_r"])
        run("age_60_85", t["sens"], cfg.PRIMARY_MEASURE, fc, metric, ["age_z", "age_z2"])
    table = pd.DataFrame(rows)
    return _fdr(table, "group_p", "group_p_fdr", by="family")


def resampling(t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for draw, cohort in t["resamples"].groupby("draw"):
        for fc, metric in PRIMARY_ENDPOINTS:
            d = network_long(t, cohort, cfg.PRIMARY_MEASURE, fc)
            res = fit_lmm(d, metric, interaction=False)
            rows.append({"draw": draw, "fc": fc, "metric": metric, **res})
    table = pd.DataFrame(rows)
    summary = []
    for (fc, metric), sub in table.groupby(["fc", "metric"], sort=False):
        row = {"fc": fc, "metric": metric, "draws": len(sub),
               "group_p_median": sub["group_p"].median(), "group_sig_fraction": (sub["group_p"] < 0.05).mean()}
        for hi, lo in CONTRASTS:
            key = f"{hi}_vs_{lo}"
            row[f"{key}_est_median"] = sub[f"{key}_est"].median()
            row[f"{key}_est_q025"] = sub[f"{key}_est"].quantile(0.025)
            row[f"{key}_est_q975"] = sub[f"{key}_est"].quantile(0.975)
            row[f"{key}_sig_fraction"] = (sub[f"{key}_p"] < 0.05).mean()
        summary.append(row)
    return table, pd.DataFrame(summary)


def ce_vs_betweenness(t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fc in ("LOFC", "HOFC", "sHOFC"):
        d = network_long(t, t["matched"], cfg.PRIMARY_MEASURE, fc)
        for band in cfg.BAND_NAMES + ["all"]:
            sub = d if band == "all" else d[d["band"] == band]
            if band == "all":
                sub = sub.groupby(["serial", "group"], as_index=False)[
                    ["AttackAUC_ce_eff", "AttackAUC_betweenness_eff", "HubJaccard_CE_BC"]].mean()
            diff = sub["AttackAUC_ce_eff"] - sub["AttackAUC_betweenness_eff"]
            bound = 0.2 * sub["AttackAUC_betweenness_eff"].std(ddof=1)
            n = diff.size
            se = diff.std(ddof=1) / np.sqrt(n)
            p_lower = stats.t.sf((diff.mean() + bound) / se, n - 1)
            p_upper = stats.t.cdf((diff.mean() - bound) / se, n - 1)
            t90 = stats.t.ppf(0.95, n - 1)
            rows.append({
                "fc": fc, "band": band, "n": n, "mean_diff_ce_minus_bc": diff.mean(),
                "ci90_lo": diff.mean() - t90 * se, "ci90_hi": diff.mean() + t90 * se,
                "equivalence_bound": bound, "p_tost": max(p_lower, p_upper),
                "p_difference": stats.ttest_1samp(diff, 0.0).pvalue,
                **{f"jaccard_{g}": sub.loc[sub["group"] == g, "HubJaccard_CE_BC"].mean() for g in cfg.GROUPS},
            })
    tost = pd.DataFrame(rows)

    inter = []
    for fc in ("LOFC", "sHOFC", "HOFC"):
        d = network_long(t, t["matched"], cfg.PRIMARY_MEASURE, fc)
        long = d.melt(id_vars=["serial", "group", "age", "band"],
                      value_vars=[f"AttackAUCdiff_{o}_eff" for o in ("ce", "degree", "betweenness", "closeness", "topsis")],
                      var_name="order", value_name="v").dropna()
        long["y"] = long.groupby(["band", "order"])["v"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
        long["age_z"] = (long["age"] - long["age"].mean()) / long["age"].std(ddof=1)
        res = _fit_mixed(f"y ~ {GROUP_TERM} * C(order) + C(band) + age_z", long)
        if res is None:
            continue
        terms = [x for x in res.fe_params.index if ":" in x]
        b = res.fe_params[terms].to_numpy()
        v = res.cov_params().loc[terms, terms].to_numpy()
        chi2 = float(b @ np.linalg.pinv(v) @ b)
        inter.append({"fc": fc, "group_x_order_chi2": chi2, "df": len(terms),
                      "group_x_order_p": float(stats.chi2.sf(chi2, len(terms)))})
    return tost, pd.DataFrame(inter)


def qc_comparison(t: dict) -> pd.DataFrame:
    cohort = t["matched"][["serial", "group", "age"]]
    d = cohort.merge(t["log"], on="serial")
    stab = t["net"][(t["net"]["measure"] == cfg.PRIMARY_MEASURE)].groupby(["serial", "fc"])["Stability_r"].mean().unstack()
    stab.columns = [f"Stability_r_{c}" for c in stab.columns]
    d = d.merge(stab.reset_index(), on="serial", how="left")
    variables = ["n_bad_channels", "n_ica_excluded", "n_excluded_eye", "n_excluded_muscle", "n_brain_components",
                 "epoch_reject_fraction", "n_candidate_epochs", "selected_span_s", "emg_ratio", "kurtosis_mean",
                 "line60_ratio_raw", "ptp_median_uv"] + list(stab.columns)
    return _compare_subject_vars(d, variables, "qc")


def spectral_comparison(t: dict) -> pd.DataFrame:
    d = t["matched"][["serial", "group", "age"]].merge(t["spectral"], on="serial")
    d["log_theta_alpha"] = np.log10(d["theta_alpha_ratio"])
    variables = [f"rel_{b}" for b in cfg.BANDS] + [f"logabs_{b}" for b in cfg.BANDS] + [
        "log_theta_alpha", "iaf_peak", "iaf_cog", "ap_offset", "ap_exponent"]
    table = _compare_subject_vars(d, variables, "spectral")
    table["missing_HC_MCI_AD"] = [
        "/".join(str(int(d.loc[d["group"] == g, v].isna().sum())) for g in cfg.GROUPS) for v in table["variable"]]
    return table


def _compare_subject_vars(d: pd.DataFrame, variables: list[str], family: str) -> pd.DataFrame:
    rows = []
    for var in variables:
        if var not in d:
            continue
        groups = [d.loc[d["group"] == g, var].dropna() for g in cfg.GROUPS]
        row = {"family": family, "variable": var}
        for g, values in zip(cfg.GROUPS, groups):
            row[f"{g}_mean"] = values.mean()
            row[f"{g}_sd"] = values.std(ddof=1)
            row[f"{g}_median"] = values.median()
        try:
            row["kruskal_H"], row["kruskal_p"] = stats.kruskal(*groups)
        except ValueError:
            row["kruskal_H"], row["kruskal_p"] = np.nan, np.nan
        ols = fit_ols(d, var, ["age_z"])
        row.update({f"ols_{k}": v for k, v in ols.items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    table = _fdr(table, "kruskal_p", "kruskal_p_fdr")
    return _fdr(table, "ols_group_p", "ols_group_p_fdr")


def nonparametric_bandwise(t: dict) -> pd.DataFrame:
    rows = []
    for fc, metric in PRIMARY_ENDPOINTS:
        d = network_long(t, t["matched"], cfg.PRIMARY_MEASURE, fc)
        for band in cfg.BANDS:
            sub = d[d["band"] == band]
            groups = {g: sub.loc[sub["group"] == g, metric].dropna() for g in cfg.GROUPS}
            h, p = stats.kruskal(*groups.values())
            row = {"fc": fc, "metric": metric, "band": band, "kruskal_H": h, "kruskal_p": p}
            for hi, lo in CONTRASTS:
                row[f"{hi}_vs_{lo}_mwu_p"] = stats.mannwhitneyu(groups[hi], groups[lo]).pvalue
                row[f"{hi}_vs_{lo}_cliffs_delta"] = _cliffs_delta(groups[hi], groups[lo])
            rows.append(row)
    table = pd.DataFrame(rows)
    return _fdr(table, "kruskal_p", "kruskal_p_fdr")


def convergence_summary(t: dict) -> pd.DataFrame:
    conv = t["convergence"]
    if conv.empty:
        return conv
    d = conv.merge(t["matched"][["serial", "group"]], on="serial")
    per_subject = d.groupby(["serial", "group", "band", "n_epochs"], as_index=False)[["r_lofc", "r_hofc", "r_hofc_signed"]].mean()
    return per_subject.groupby(["band", "n_epochs", "group"])[["r_lofc", "r_hofc", "r_hofc_signed"]].agg(["mean", "std"]).reset_index()


def descriptives(t: dict) -> pd.DataFrame:
    cohort = t["matched"][["serial", "group"]]
    net = cohort.merge(t["net"], on="serial")
    metrics = [c for c in net.columns if c not in ("serial", "group", "measure", "band", "fc")]
    long = net.melt(id_vars=["group", "measure", "band", "fc"], value_vars=metrics, var_name="metric").dropna()
    return long.groupby(["measure", "fc", "metric", "band", "group"])["value"].agg(["mean", "std", "count"]).reset_index()


def main() -> None:
    parser = cfg.add_common_args(argparse.ArgumentParser(description="CAUEEG 预先指定统计分析"))
    parser.add_argument("--skip-resampling", action="store_true")
    args = parser.parse_args()
    paths = cfg.paths_from_args(args)
    t = load_tables(paths)
    save = lambda df, name: df.to_csv(paths.results / name, index=False, encoding="utf-8-sig")  # noqa: E731

    prim, bands = primary(t)
    save(prim, "stats_primary.csv")
    save(bands, "stats_bandwise.csv")
    save(power_adjusted(t, prim), "stats_power_adjusted.csv")
    save(power_adjusted_bandwise(t), "stats_power_adjusted_bandwise.csv")
    save(robustness(t), "stats_robustness.csv")
    tost, inter = ce_vs_betweenness(t)
    save(tost, "stats_ce_vs_betweenness.csv")
    save(inter, "stats_order_interaction.csv")
    save(qc_comparison(t), "stats_qc.csv")
    save(spectral_comparison(t), "stats_spectral.csv")
    save(nonparametric_bandwise(t), "stats_nonparametric_bandwise.csv")
    conv = convergence_summary(t)
    if not conv.empty:
        conv.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in conv.columns]
        save(conv, "hofc_convergence_summary.csv")
    save(descriptives(t), "group_descriptives.csv")
    if not args.skip_resampling:
        draws, summary = resampling(t)
        save(draws, "stats_resampling.csv")
        save(summary, "stats_resampling_summary.csv")

    cols = ["fc", "metric", "n_subjects", "group_chi2", "group_p", "group_p_fdr", "MCI_vs_HC_est",
            "AD_vs_HC_est", "AD_vs_MCI_est", "interaction_p", "interaction_p_fdr"]
    print(prim[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
