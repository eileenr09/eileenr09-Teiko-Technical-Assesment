from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / "cell_counts.db"
OUTPUT_DIR = ROOT / "outputs"
POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
RANDOM_STATE = 42


# Part 3 statistics helpers

def rank_biserial_effect_size(u_statistic: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation derived from the Mann-Whitney U statistic.

    Ranges from -1 to 1; magnitude is the effect size, sign indicates which
    group (responders, here) tends to rank higher.
    """
    return 1 - (2 * u_statistic) / (n1 * n2)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction, returned in the original order.

    Needed because five populations are tested at once (Part 3); reporting
    raw p-values alone overstates significance under repeated testing.
    """
    n = len(p_values)
    sorted_p = p_values.to_numpy()[np.argsort(p_values.to_numpy())]
    ranks = np.arange(1, n + 1)
    q = sorted_p * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    unsorted_q = np.empty(n)
    unsorted_q[np.argsort(p_values.to_numpy())] = q
    return pd.Series(unsorted_q, index=p_values.index)


def bootstrap_mean_diff_ci(
    responders: np.ndarray, non_responders: np.ndarray, n_boot: int = 10000
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the responder-minus-non-responder mean gap."""
    rng = np.random.default_rng(RANDOM_STATE)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        r_sample = rng.choice(responders, size=len(responders), replace=True)
        n_sample = rng.choice(non_responders, size=len(non_responders), replace=True)
        diffs[i] = r_sample.mean() - n_sample.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """L2-regularized logistic regression fit by direct likelihood maximization.

    A hand-rolled fit (via scipy.optimize) rather than a library model: this
    keeps the signal-model baseline dependency-free (numpy/scipy only, both
    already required for the stats above) so it runs the same way on any
    machine that can already run the rest of the pipeline.
    """
    n_samples, n_features = X.shape
    design = np.hstack([np.ones((n_samples, 1)), X])

    def neg_log_likelihood(beta: np.ndarray) -> float:
        z = design @ beta
        log_lik = np.sum(y * z - np.logaddexp(0, z))
        penalty = 0.5 * l2 * np.sum(beta[1:] ** 2)
        return -log_lik + penalty

    def gradient(beta: np.ndarray) -> np.ndarray:
        z = design @ beta
        p_hat = 1.0 / (1.0 + np.exp(-z))
        grad = design.T @ (p_hat - y)
        grad[1:] += l2 * beta[1:]
        return grad

    result = minimize(
        neg_log_likelihood,
        x0=np.zeros(n_features + 1),
        jac=gradient,
        method="L-BFGS-B",
    )
    return result.x


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((X.shape[0], 1)), X])
    z = design @ beta
    return 1.0 / (1.0 + np.exp(-z))


def roc_curve_manual(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    positives = y_true.sum()
    negatives = len(y_true) - positives

    true_positives = np.cumsum(y_sorted)
    false_positives = np.cumsum(1 - y_sorted)
    threshold_ends = np.r_[np.where(np.diff(scores_sorted) != 0)[0], len(scores_sorted) - 1]

    tpr = np.r_[0, true_positives[threshold_ends] / positives]
    fpr = np.r_[0, false_positives[threshold_ends] / negatives]
    return fpr, tpr


def subject_grouped_stratified_folds(
    subject_response: pd.Series, n_splits: int, seed: int
) -> dict:
    """Assign every subject to one of n_splits folds, balanced by response.

    Grouped by subject (all of a subject's samples land in the same fold) and
    stratified by that subject's response, so cross-validated AUC reflects
    generalization to new patients rather than leaking a subject's own
    signal from train into test.
    """
    rng = np.random.default_rng(seed)
    fold_of_subject = {}
    for response_value in subject_response.unique():
        ids = subject_response.index[subject_response == response_value].to_numpy()
        ids = rng.permutation(ids)
        for position, subject_id in enumerate(ids):
            fold_of_subject[subject_id] = position % n_splits
    return fold_of_subject


def permutation_test_auc(
    X_scaled: np.ndarray,
    sample_subject_ids: np.ndarray,
    subject_response: pd.Series,
    fold_of_subject: dict,
    n_splits: int,
    observed_auc: float,
    n_permutations: int = 200,
    seed: int = RANDOM_STATE,
) -> tuple[float, np.ndarray]:
    """Empirical p-value for the observed AUC against a label-shuffled null.

    An AUC of 0.54 could still just be noise at this sample size. This
    reshuffles which subjects are labeled responders (same fold assignment,
    same class balance) and reruns the identical CV fit each time, building a
    null distribution of AUCs under "no real association" between the cell
    mix and response - the fraction of null AUCs at or above the observed one
    is the p-value.
    """
    rng = np.random.default_rng(seed)
    subject_ids = subject_response.index.to_numpy()
    sample_fold = np.array([fold_of_subject[s] for s in sample_subject_ids])
    null_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = pd.Series(rng.permutation(subject_response.to_numpy()), index=subject_ids)
        y_perm = pd.Series(sample_subject_ids).map(shuffled).eq("yes").astype(int).to_numpy()
        oof_perm = np.empty(len(y_perm))
        for fold in range(n_splits):
            test_mask = sample_fold == fold
            train_mask = ~test_mask
            beta = fit_logistic_regression(X_scaled[train_mask], y_perm[train_mask])
            oof_perm[test_mask] = predict_proba(X_scaled[test_mask], beta)
        fpr_p, tpr_p = roc_curve_manual(y_perm, oof_perm)
        null_aucs[i] = np.trapezoid(tpr_p, fpr_p)
    p_value = (1 + np.sum(null_aucs >= observed_auc)) / (n_permutations + 1)
    return float(p_value), null_aucs


def clr_transform(proportions: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio transform.

    Cell-population percentages are compositional (they sum to 100 per
    sample), so raw percentages are not independent and violate the
    assumptions behind Euclidean-geometry methods like PCA/logistic
    regression (a rise in one population mechanically pushes the others
    down). CLR maps the simplex onto real space so those methods, and the
    correlations/PCA below, are not just measuring the shared constraint.
    """
    eps = 1e-6
    clipped = proportions.clip(lower=eps)
    log_vals = np.log(clipped)
    return log_vals.sub(log_vals.mean(axis=1), axis=0)


# Main pipeline

def run_analysis() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        summary = pd.read_sql_query(
            "SELECT * FROM sample_population_summary ORDER BY sample, population",
            connection,
        )
        comparison = pd.read_sql_query(
            """
            SELECT sample, subject_id, population, percentage, response
            FROM sample_population_summary
            WHERE condition = 'melanoma'
              AND treatment = 'miraclib'
              AND sample_type = 'PBMC'
              AND response IN ('yes', 'no')
            """,
            connection,
        )
        baseline = pd.read_sql_query(
            """
            SELECT s.sample_id AS sample, subject_id, sex, age, project, treatment, response,
                   time_from_treatment_start, b.count AS b_cell_count
            FROM samples AS s
            JOIN cell_counts AS b
              ON b.sample_id = s.sample_id AND b.population = 'b_cell'
            WHERE s.condition = 'melanoma'
              AND s.sample_type = 'PBMC'
              AND s.treatment = 'miraclib'
              AND s.time_from_treatment_start = 0
            ORDER BY s.sample_id
            """,
            connection,
        )
        answer = pd.read_sql_query(
            """
            SELECT AVG(b.count) AS average_b_cell_count
            FROM samples AS s
            JOIN cell_counts AS b
              ON b.sample_id = s.sample_id AND b.population = 'b_cell'
            WHERE s.condition = 'melanoma'
              AND s.sex = 'M'
              AND s.response = 'yes'
              AND s.time_from_treatment_start = 0
            """,
            connection,
        ).iloc[0, 0]
        cohort_subject_balance = pd.read_sql_query(
            """
            SELECT DISTINCT subject_id, sex, age, response
            FROM samples
            WHERE condition = 'melanoma'
              AND treatment = 'miraclib'
              AND sample_type = 'PBMC'
              AND response IN ('yes', 'no')
            """,
            connection,
        )
        longitudinal_source = pd.read_sql_query(
            """
            SELECT sample, subject_id, population, percentage, response,
                   treatment, time_from_treatment_start
            FROM sample_population_summary
            WHERE condition = 'melanoma'
              AND sample_type = 'PBMC'
              AND treatment IN ('miraclib', 'phauximab')
              AND response IN ('yes', 'no')
            """,
            connection,
        )

    # Part 2 output
    summary.to_csv(OUTPUT_DIR / "population_summary.csv", index=False)

    # Cohort overview, full trial table
    sample_meta = summary.drop_duplicates("sample")
    cohort_rows = []
    for dimension in ("condition", "treatment", "sample_type", "sex", "project"):
        grouped = sample_meta.groupby(dimension).agg(
            sample_count=("sample", "count"),
            subject_count=("subject_id", "nunique"),
        )
        for category, row in grouped.iterrows():
            cohort_rows.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "sample_count": int(row["sample_count"]),
                    "subject_count": int(row["subject_count"]),
                }
            )
    pd.DataFrame(cohort_rows).to_csv(OUTPUT_DIR / "cohort_overview.csv", index=False)

    subject_demographics = (
        sample_meta.groupby("subject_id")
        .agg(
            project=("project", "first"),
            condition=("condition", "first"),
            treatment=("treatment", "first"),
            sex=("sex", "first"),
            age=("age", "first"),
            response=("response", "first"),
        )
        .reset_index()
    )
    subject_demographics.to_csv(OUTPUT_DIR / "subject_demographics.csv", index=False)

    # Part 3 responder vs non-responder stats
    results = []
    for population in POPULATIONS:
        values = comparison[comparison["population"] == population]
        responders = values.loc[values["response"] == "yes", "percentage"].to_numpy()
        non_responders = values.loc[values["response"] == "no", "percentage"].to_numpy()
        statistic, p_value = mannwhitneyu(
            responders, non_responders, alternative="two-sided"
        )
        ci_low, ci_high = bootstrap_mean_diff_ci(responders, non_responders)
        results.append(
            {
                "population": population,
                "responders_n": len(responders),
                "non_responders_n": len(non_responders),
                "responders_mean_percentage": responders.mean(),
                "non_responders_mean_percentage": non_responders.mean(),
                "difference_percentage_points": responders.mean() - non_responders.mean(),
                "diff_ci_low_95": ci_low,
                "diff_ci_high_95": ci_high,
                "rank_biserial_effect_size": rank_biserial_effect_size(
                    statistic, len(responders), len(non_responders)
                ),
                "mann_whitney_u": statistic,
                "p_value": p_value,
            }
        )

    results_df = pd.DataFrame(results)
    results_df["q_value_fdr_bh"] = benjamini_hochberg(results_df["p_value"])
    results_df["significant_at_q0.05"] = results_df["q_value_fdr_bh"] < 0.05
    comparison.to_csv(OUTPUT_DIR / "response_comparison.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "statistical_results.csv", index=False)

    # Covariate balance, rule out age and sex confounding
    resp_age = cohort_subject_balance.loc[cohort_subject_balance["response"] == "yes", "age"]
    nonresp_age = cohort_subject_balance.loc[cohort_subject_balance["response"] == "no", "age"]
    age_u, age_p = mannwhitneyu(resp_age, nonresp_age, alternative="two-sided")

    sex_table = pd.crosstab(cohort_subject_balance["sex"], cohort_subject_balance["response"])
    if sex_table.shape == (2, 2):
        sex_odds_ratio, sex_p = fisher_exact(sex_table.to_numpy())
    else:
        sex_odds_ratio, sex_p = float("nan"), float("nan")

    balance_rows = [
        {
            "covariate": "age",
            "test": "Mann-Whitney U",
            "statistic": age_u,
            "p_value": age_p,
            "responders_summary": f"mean={resp_age.mean():.1f}, n={len(resp_age)}",
            "non_responders_summary": f"mean={nonresp_age.mean():.1f}, n={len(nonresp_age)}",
        },
        {
            "covariate": "sex",
            "test": "Fisher exact (2x2)",
            "statistic": sex_odds_ratio,
            "p_value": sex_p,
            "responders_summary": ", ".join(f"{sex}={sex_table.loc[sex, 'yes']}" for sex in sex_table.index),
            "non_responders_summary": ", ".join(f"{sex}={sex_table.loc[sex, 'no']}" for sex in sex_table.index),
        },
    ]
    pd.DataFrame(balance_rows).to_csv(OUTPUT_DIR / "cohort_balance.csv", index=False)

    # Feature extraction, ratios and compositional geometry
    wide_pct = comparison.pivot_table(
        index=["sample", "subject_id", "response"], columns="population", values="percentage"
    ).reset_index()[["sample", "subject_id", "response", *POPULATIONS]]

    derived = wide_pct[["sample", "subject_id", "response"]].copy()
    derived["cd4_cd8_ratio"] = wide_pct["cd4_t_cell"] / wide_pct["cd8_t_cell"]
    derived["nk_monocyte_ratio"] = wide_pct["nk_cell"] / wide_pct["monocyte"]
    lymphoid = wide_pct["b_cell"] + wide_pct["cd4_t_cell"] + wide_pct["cd8_t_cell"] + wide_pct["nk_cell"]
    derived["lymphoid_myeloid_ratio"] = lymphoid / wide_pct["monocyte"]
    derived.to_csv(OUTPUT_DIR / "derived_features.csv", index=False)

    proportions = wide_pct[POPULATIONS] / 100.0
    clr = clr_transform(proportions)
    clr.columns = [f"clr_{c}" for c in clr.columns]

    correlations = proportions.corr(method="spearman")
    correlations.to_csv(OUTPUT_DIR / "population_correlations.csv")

    # Longitudinal trends, population mix over time
    trend_summary = (
        longitudinal_source.groupby(["population", "treatment", "response", "time_from_treatment_start"])[
            "percentage"
        ]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_percentage", "count": "n"})
    )
    trend_summary["ci_low"] = trend_summary["mean_percentage"] - 1.96 * trend_summary["sem"]
    trend_summary["ci_high"] = trend_summary["mean_percentage"] + 1.96 * trend_summary["sem"]
    trend_summary.to_csv(OUTPUT_DIR / "longitudinal_trends.csv", index=False)

    longitudinal_stats_frames = []
    for treatment in ("miraclib", "phauximab"):
        arm = longitudinal_source[longitudinal_source["treatment"] == treatment]
        arm_rows = []
        for population in POPULATIONS:
            for timepoint in sorted(arm["time_from_treatment_start"].unique()):
                cell = arm[(arm["population"] == population) & (arm["time_from_treatment_start"] == timepoint)]
                r = cell.loc[cell["response"] == "yes", "percentage"].to_numpy()
                nr = cell.loc[cell["response"] == "no", "percentage"].to_numpy()
                u_stat, p_val = mannwhitneyu(r, nr, alternative="two-sided")
                arm_rows.append(
                    {
                        "treatment": treatment,
                        "population": population,
                        "time_from_treatment_start": timepoint,
                        "responders_n": len(r),
                        "non_responders_n": len(nr),
                        "responders_mean": r.mean(),
                        "non_responders_mean": nr.mean(),
                        "mann_whitney_u": u_stat,
                        "p_value": p_val,
                    }
                )
        arm_df = pd.DataFrame(arm_rows)
        arm_df["q_value_fdr_bh"] = benjamini_hochberg(arm_df["p_value"])
        longitudinal_stats_frames.append(arm_df)

    longitudinal_stats = pd.concat(longitudinal_stats_frames, ignore_index=True)
    longitudinal_stats["significant_at_q0.05"] = longitudinal_stats["q_value_fdr_bh"] < 0.05
    longitudinal_stats.to_csv(OUTPUT_DIR / "longitudinal_stats.csv", index=False)

    # Signal model, can population mix predict response
    X = clr.to_numpy()
    X_scaled = standardize(X)
    y = (wide_pct["response"] == "yes").astype(int).to_numpy()

    subject_response = wide_pct.drop_duplicates("subject_id").set_index("subject_id")["response"]
    n_splits = 5
    fold_of_subject = subject_grouped_stratified_folds(subject_response, n_splits, RANDOM_STATE)
    sample_fold = wide_pct["subject_id"].map(fold_of_subject).to_numpy()

    oof_proba = np.empty(len(y))
    fold_aucs = []
    for fold in range(n_splits):
        test_mask = sample_fold == fold
        train_mask = ~test_mask
        beta = fit_logistic_regression(X_scaled[train_mask], y[train_mask])
        fold_proba = predict_proba(X_scaled[test_mask], beta)
        oof_proba[test_mask] = fold_proba
        fold_fpr, fold_tpr = roc_curve_manual(y[test_mask], fold_proba)
        fold_aucs.append(float(np.trapezoid(fold_tpr, fold_fpr)))

    fpr, tpr = roc_curve_manual(y, oof_proba)
    oof_auc = float(np.trapezoid(tpr, fpr))
    oof_accuracy = float(np.mean((oof_proba >= 0.5) == y))

    auc_p_value, null_aucs = permutation_test_auc(
        X_scaled,
        wide_pct["subject_id"].to_numpy(),
        subject_response,
        fold_of_subject,
        n_splits,
        oof_auc,
        n_permutations=200,
    )
    pd.DataFrame({"null_auc": null_aucs}).to_csv(
        OUTPUT_DIR / "signal_model_permutation_null.csv", index=False
    )

    final_beta = fit_logistic_regression(X_scaled, y)
    coefficients = pd.DataFrame(
        {
            "population": POPULATIONS,
            "standardized_coefficient": final_beta[1:],
            "odds_ratio": np.exp(final_beta[1:]),
        }
    ).sort_values("standardized_coefficient", key=np.abs, ascending=False)

    centered = X_scaled - X_scaled.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ vt[:2].T
    explained_variance_ratio = (singular_values**2 / np.sum(singular_values**2))[:2]
    pca_df = wide_pct[["sample", "subject_id", "response"]].copy()
    pca_df["pc1"] = pcs[:, 0]
    pca_df["pc2"] = pcs[:, 1]

    pca_loadings = pd.DataFrame(
        {"population": POPULATIONS, "pc1_loading": vt[0], "pc2_loading": vt[1]}
    )

    oof_predictions = wide_pct[["sample", "subject_id", "response"]].copy()
    oof_predictions["oof_probability"] = oof_proba
    oof_predictions["predicted_response"] = np.where(oof_proba >= 0.5, "yes", "no")
    oof_predictions["correct"] = oof_predictions["predicted_response"] == oof_predictions["response"]

    pd.DataFrame(
        {
            "cv_fold": range(1, len(fold_aucs) + 1),
            "fold_auc": fold_aucs,
        }
    ).to_csv(OUTPUT_DIR / "signal_model_cv_folds.csv", index=False)

    pd.DataFrame(
        [
            {
                "n_samples": len(y),
                "n_subjects": len(subject_response),
                "n_responders": int(y.sum()),
                "n_non_responders": int(len(y) - y.sum()),
                "cv_scheme": "5-fold, grouped and stratified by subject_id",
                "features": "CLR-transformed population proportions",
                "oof_roc_auc": oof_auc,
                "oof_accuracy": oof_accuracy,
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "std_fold_auc": float(np.std(fold_aucs)),
                "auc_permutation_p_value": auc_p_value,
                "n_permutations": 200,
            }
        ]
    ).to_csv(OUTPUT_DIR / "signal_model_metrics.csv", index=False)

    coefficients.to_csv(OUTPUT_DIR / "signal_model_coefficients.csv", index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(OUTPUT_DIR / "signal_model_roc_curve.csv", index=False)
    pca_df.to_csv(OUTPUT_DIR / "signal_model_pca.csv", index=False)
    pd.DataFrame(
        {"component": ["PC1", "PC2"], "explained_variance_ratio": explained_variance_ratio}
    ).to_csv(OUTPUT_DIR / "signal_model_pca_variance.csv", index=False)
    pca_loadings.to_csv(OUTPUT_DIR / "signal_model_pca_loadings.csv", index=False)
    oof_predictions.to_csv(OUTPUT_DIR / "signal_model_oof_predictions.csv", index=False)

    # Part 4 baseline subset and required breakdowns
    baseline.to_csv(OUTPUT_DIR / "baseline_melanoma_pbmc_miraclib.csv", index=False)

    subjects = baseline.drop_duplicates("subject_id")
    breakdown_rows = []
    for project, count in baseline["project"].value_counts().items():
        breakdown_rows.append({"breakdown": "samples_per_project", "category": project, "count": int(count)})
    for response, count in subjects["response"].value_counts().items():
        breakdown_rows.append({"breakdown": "responders_by_subject", "category": response, "count": int(count)})
    for sex, count in subjects["sex"].value_counts().items():
        breakdown_rows.append({"breakdown": "sex_by_subject", "category": sex, "count": int(count)})
    pd.DataFrame(breakdown_rows).to_csv(OUTPUT_DIR / "subset_breakdown.csv", index=False)

    (OUTPUT_DIR / "answer.txt").write_text(
        f"Average B-cell count: {answer:.2f}\n", encoding="utf-8"
    )

    significant = results_df.loc[results_df["significant_at_q0.05"], "population"].tolist()
    longitudinal_significant = longitudinal_stats.loc[
        longitudinal_stats["significant_at_q0.05"], ["treatment", "population", "time_from_treatment_start"]
    ]
    print(f"Wrote analysis outputs to {OUTPUT_DIR}")
    print(f"Significant populations at FDR q<0.05 (pooled across timepoints): {', '.join(significant) or 'none'}")
    print(f"Age balance responders vs non-responders: Mann-Whitney p={age_p:.3f}")
    print(f"Sex balance responders vs non-responders: Fisher exact p={sex_p:.3f}")
    if len(longitudinal_significant):
        print(f"Timepoint-specific significant findings:\n{longitudinal_significant.to_string(index=False)}")
    else:
        print("No population is significant at any single timepoint after FDR correction, either.")
    print(
        f"Signal model out-of-fold ROC-AUC: {oof_auc:.3f} (mean fold AUC {np.mean(fold_aucs):.3f}), "
        f"permutation p={auc_p_value:.3f}"
    )
    print(f"Melanoma male responder baseline average B-cell count: {answer:.2f}")


if __name__ == "__main__":
    run_analysis()
