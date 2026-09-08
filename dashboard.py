from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / "cell_counts.db"
OUTPUT_DIR = ROOT / "outputs"
POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

# Fixed categorical hues (validated for colorblind-safety); assigned by entity
# identity and reused across every chart, never re-cycled per filter.
POPULATION_COLORS = {
    "b_cell": "#2a78d6",
    "cd8_t_cell": "#eb6834",
    "cd4_t_cell": "#1baf7a",
    "nk_cell": "#eda100",
    "monocyte": "#e87ba4",
}
RESPONSE_COLORS = {"yes": "#2a78d6", "no": "#eb6834"}
DIVERGING_SCALE = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
COUNT_BAR_COLOR = "#2a78d6"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


st.set_page_config(page_title="Loblaw Bio immune analysis", layout="wide")
st.title("Loblaw Bio | Immune response analysis")
st.caption(
    "Bob Loblaw, a drug developer at Loblaw Bio, is running a clinical trial and "
    "needs to understand how his drug candidate (miraclib) affects five immune "
    "cell populations (b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte) across "
    "patient samples. This dashboard presents that analysis end to end: population "
    "frequencies, the responder-vs-non-responder comparison, a baseline subset "
    "breakdown, and two bonus views on trends over time and predictive signal."
)

if not DATABASE_PATH.exists():
    st.error("Database not found. Run `make pipeline` first.")
    st.stop()


@st.cache_data
def load_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(query, connection)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


summary = load_sql("SELECT * FROM sample_population_summary")
baseline = load_sql(
    """
    SELECT s.sample_id AS sample, s.subject_id, s.sex, s.age, s.project, s.treatment,
           s.response, s.time_from_treatment_start, c.count AS b_cell_count
    FROM samples s JOIN cell_counts c ON c.sample_id = s.sample_id
    WHERE c.population = 'b_cell' AND s.condition = 'melanoma'
      AND s.sample_type = 'PBMC' AND s.treatment = 'miraclib'
      AND s.time_from_treatment_start = 0
    ORDER BY s.sample_id
    """
)

comparison = load_csv("response_comparison.csv")
stats_results = load_csv("statistical_results.csv")
cohort_balance = load_csv("cohort_balance.csv")
derived_features = load_csv("derived_features.csv")
correlations = load_csv("population_correlations.csv")
subset_breakdown = load_csv("subset_breakdown.csv")
answer_path = OUTPUT_DIR / "answer.txt"
signal_metrics = load_csv("signal_model_metrics.csv")
signal_coefficients = load_csv("signal_model_coefficients.csv")
signal_roc = load_csv("signal_model_roc_curve.csv")
signal_pca = load_csv("signal_model_pca.csv")
signal_pca_variance = load_csv("signal_model_pca_variance.csv")
signal_pca_loadings = load_csv("signal_model_pca_loadings.csv")
signal_cv_folds = load_csv("signal_model_cv_folds.csv")
signal_oof_predictions = load_csv("signal_model_oof_predictions.csv")
signal_permutation_null = load_csv("signal_model_permutation_null.csv")
cohort_overview = load_csv("cohort_overview.csv")
subject_demographics = load_csv("subject_demographics.csv")
longitudinal_trends = load_csv("longitudinal_trends.csv")
longitudinal_stats = load_csv("longitudinal_stats.csv")

with st.expander("Database schema (Part 1)"):
    st.markdown(
        "`cell_counts.db` has two tables and a view:\n\n"
        "- **`samples`**: one row per biological sample, holding sample and subject "
        "metadata (`project`, `subject_id`, `condition`, `age`, `sex`, `treatment`, "
        "`response`, `sample_type`, `time_from_treatment_start`).\n"
        "- **`cell_counts`**: one row per `(sample_id, population)` pair with the "
        "raw count, foreign keyed to `samples`. Storing populations long rather "
        "than as five separate columns means a new population is a data change, "
        "not a schema migration, and every aggregate is a `GROUP BY` rather than a "
        "hardcoded column list.\n"
        "- **`sample_population_summary`** (view): joins the two tables and "
        "computes `total_count` and `percentage` per row, so Part 2's table is "
        "just `SELECT * FROM sample_population_summary`, computed once rather "
        "than reimplemented in every consumer.\n\n"
        "**Rationale and scaling.** Subject metadata is technically repeated "
        "across a subject's samples, but at hundreds of projects and thousands of "
        "samples that duplication is small next to the benefit of keeping every "
        "query a single join. At real scale, the next step is normalizing "
        "`subjects` and `projects` into their own tables, with `samples` holding "
        "only sample-specific fields and foreign keys. The long `cell_counts` "
        "layout already supports new population types with no schema change, and "
        "composes cleanly with a `panels`/`markers` table if the assay ever "
        "reports per-marker values instead of discrete populations. For query "
        "performance at scale, indexes on "
        "`samples(condition, treatment, sample_type, time_from_treatment_start)` "
        "and `cell_counts(population)` would be the first additions, since those "
        "are the columns every analysis here filters or groups on."
    )

with st.expander("Code structure"):
    st.markdown(
        "- **`load_data.py`** (Part 1): validates the CSV, rebuilds the schema, "
        "and loads it into SQLite. Fails loudly on bad input (missing columns, "
        "duplicate sample IDs, negative or all-zero counts) instead of loading "
        "partial data.\n"
        "- **`analysis.py`**: runs Parts 2 to 4 plus the longitudinal and "
        "signal-model extras, reading from the database and writing every result "
        "to `outputs/`. It takes no arguments, so `make pipeline` is just "
        "`python load_data.py` followed by `python analysis.py`.\n"
        "- **`dashboard.py`** (this app): a read-only Streamlit view over "
        "`cell_counts.db` and `outputs/*.csv`. It never recomputes statistics "
        "itself, it only renders what `analysis.py` already wrote, so the "
        "dashboard can never disagree with the output files."
    )

with st.expander("Generated outputs reference"):
    st.markdown(
        "`make pipeline` writes `cell_counts.db` and these files under "
        "`outputs/`:\n\n"
        "- `population_summary.csv`, `cohort_overview.csv`, "
        "`subject_demographics.csv`\n"
        "- `response_comparison.csv`, `statistical_results.csv`, "
        "`cohort_balance.csv`\n"
        "- `derived_features.csv`, `population_correlations.csv`\n"
        "- `longitudinal_trends.csv`, `longitudinal_stats.csv`\n"
        "- `signal_model_metrics.csv`, `signal_model_cv_folds.csv`, "
        "`signal_model_permutation_null.csv`, `signal_model_coefficients.csv`, "
        "`signal_model_roc_curve.csv`, `signal_model_pca.csv`, "
        "`signal_model_pca_variance.csv`, `signal_model_pca_loadings.csv`, "
        "`signal_model_oof_predictions.csv`\n"
        "- `baseline_melanoma_pbmc_miraclib.csv`, `subset_breakdown.csv`, "
        "`answer.txt`"
    )

tab_cohort, tab_overview, tab_response, tab_longitudinal, tab_subset, tab_signal = st.tabs(
    [
        "Cohort overview",
        "Population overview",
        "Response comparison",
        "Longitudinal trends",
        "Subset analysis",
        "Signal model",
    ]
)

# Cohort overview, trial wide summary
with tab_cohort:
    st.subheader("Trial cohort at a glance")
    st.caption(
        "Every sample and subject in cell-count.csv, before any Part 3/4 filtering - "
        "context for the specific comparisons in the other tabs."
    )

    if subject_demographics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", f"{len(summary['sample'].unique()):,}")
        c2.metric("Subjects", f"{len(subject_demographics):,}")
        c3.metric("Projects", subject_demographics["project"].nunique())
        c4.metric("Indications", subject_demographics["condition"].nunique())

    if cohort_overview is not None:
        dim_cols = st.columns(4)
        for col, dimension, title in zip(
            dim_cols,
            ["condition", "treatment", "sample_type", "project"],
            ["By indication", "By treatment", "By sample type", "By project"],
        ):
            rows = cohort_overview[cohort_overview["dimension"] == dimension].sort_values(
                "subject_count", ascending=True
            )
            fig = px.bar(
                rows,
                x="subject_count",
                y="category",
                orientation="h",
                text="subject_count",
                labels={"subject_count": "Subjects", "category": ""},
                title=title,
            )
            fig.update_traces(marker_color=COUNT_BAR_COLOR)
            fig.update_layout(showlegend=False, height=260, margin=dict(t=40, b=20))
            with col:
                st.plotly_chart(fig, width="stretch")

    if subject_demographics is not None:
        st.subheader("Age distribution")
        fig_age = px.histogram(
            subject_demographics,
            x="age",
            color="sex",
            barmode="overlay",
            opacity=0.7,
            nbins=20,
            color_discrete_map={"M": "#2a78d6", "F": "#eb6834"},
            labels={"age": "Age", "sex": "Sex"},
        )
        st.plotly_chart(fig_age, width="stretch")

# Part 2 population overview
with tab_overview:
    st.subheader("Relative frequency by sample")
    st.caption("Each sample's five populations sum to 100% of that sample's total cell count.")

    mean_freq = (
        summary.groupby("population", as_index=False)["percentage"].mean()
        .assign(population=lambda d: pd.Categorical(d["population"], POPULATIONS, ordered=True))
        .sort_values("population")
    )
    fig_mean = px.bar(
        mean_freq,
        x="population",
        y="percentage",
        color="population",
        color_discrete_map=POPULATION_COLORS,
        text_auto=".1f",
        labels={"percentage": "Mean relative frequency (%)", "population": "Population"},
        title="Average population share across all samples",
    )
    fig_mean.update_layout(showlegend=False)
    st.plotly_chart(fig_mean, width='stretch')

    selected_samples = st.multiselect(
        "Sample filter", sorted(summary["sample"].unique()), default=[]
    )
    shown = summary[summary["sample"].isin(selected_samples)] if selected_samples else summary
    st.dataframe(
        shown[["sample", "total_count", "population", "count", "percentage"]],
        width='stretch',
        hide_index=True,
    )

# Part 3 responder vs non-responder comparison
with tab_response:
    st.subheader("Miraclib melanoma PBMC: responders vs. non-responders")
    if comparison is not None:
        fig_box = px.box(
            comparison,
            x="population",
            y="percentage",
            color="response",
            points="outliers",
            category_orders={"population": POPULATIONS, "response": ["yes", "no"]},
            color_discrete_map=RESPONSE_COLORS,
            labels={"percentage": "Relative frequency (%)", "response": "Response"},
        )
        st.plotly_chart(fig_box, width='stretch')

    if stats_results is not None:
        st.markdown(
            "**Mann-Whitney U test per population**, with a rank-biserial effect size, "
            "a 95% bootstrap CI on the mean gap, and a Benjamini-Hochberg FDR-adjusted "
            "q-value (five populations are tested at once, so the raw p-values alone "
            "overstate significance)."
        )
        display_cols = [
            "population", "responders_n", "non_responders_n",
            "responders_mean_percentage", "non_responders_mean_percentage",
            "difference_percentage_points", "diff_ci_low_95", "diff_ci_high_95",
            "rank_biserial_effect_size", "p_value", "q_value_fdr_bh", "significant_at_q0.05",
        ]
        st.dataframe(
            stats_results[display_cols].round(4), width='stretch', hide_index=True
        )
        significant = stats_results.loc[stats_results["significant_at_q0.05"], "population"]
        if len(significant):
            st.success(f"Significant after FDR correction (q < 0.05): {', '.join(significant)}")
        else:
            st.info(
                "No population's relative frequency survives FDR correction at q < 0.05. "
                "cd4_t_cell has the smallest raw p-value but does not hold up once tested "
                "against the other four populations jointly - this is why the signal-model "
                "tab looks at the whole population mix rather than one cell type at a time."
            )

    if cohort_balance is not None:
        with st.expander("Baseline covariate balance check (age, sex)"):
            st.caption(
                "Before trusting a cell-population difference, check the two groups aren't "
                "just different in age or sex composition - that would confound any claim "
                "about immune populations predicting response."
            )
            st.dataframe(cohort_balance, width="stretch", hide_index=True)
            if (cohort_balance["p_value"] > 0.05).all():
                st.success("No evidence of age or sex imbalance between responders and non-responders.")
            else:
                st.warning("At least one covariate is imbalanced - treat population differences with caution.")

# Longitudinal trends, population mix over time
with tab_longitudinal:
    st.subheader("Population trends over time, responders vs. non-responders")
    st.caption(
        "Melanoma PBMC samples at days 0, 7, and 14. Bands are 95% CIs (mean +/- 1.96 x SEM). "
        "Pick a treatment arm to compare the trial drug against the comparator."
    )

    if longitudinal_trends is not None:
        treatment_choice = st.radio(
            "Treatment arm", ["miraclib", "phauximab"], horizontal=True
        )
        trend_data = longitudinal_trends[longitudinal_trends["treatment"] == treatment_choice]

        fig = make_subplots(
            rows=2, cols=3, subplot_titles=POPULATIONS,
            shared_xaxes=True,
        )
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        legend_shown = set()
        for population, (row, col) in zip(POPULATIONS, positions):
            for response_value in ("yes", "no"):
                series = trend_data[
                    (trend_data["population"] == population) & (trend_data["response"] == response_value)
                ].sort_values("time_from_treatment_start")
                if series.empty:
                    continue
                color = RESPONSE_COLORS[response_value]
                fig.add_trace(
                    go.Scatter(
                        x=series["time_from_treatment_start"], y=series["ci_high"],
                        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=series["time_from_treatment_start"], y=series["ci_low"],
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=hex_to_rgba(color, 0.18), showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=col,
                )
                show_legend = response_value not in legend_shown
                legend_shown.add(response_value)
                fig.add_trace(
                    go.Scatter(
                        x=series["time_from_treatment_start"], y=series["mean_percentage"],
                        mode="lines+markers", line=dict(color=color, width=2),
                        name=f"Response: {response_value}", legendgroup=response_value,
                        showlegend=show_legend,
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(tickvals=[0, 7, 14], title_text="Day")
        fig.update_yaxes(title_text="Mean %")
        fig.update_layout(height=560, title=f"{treatment_choice} - population trends by response")
        st.plotly_chart(fig, width="stretch")

        if longitudinal_stats is not None:
            arm_stats = longitudinal_stats[longitudinal_stats["treatment"] == treatment_choice]
            significant_points = arm_stats.loc[
                arm_stats["significant_at_q0.05"],
                ["population", "time_from_treatment_start", "responders_mean", "non_responders_mean", "q_value_fdr_bh"],
            ]
            st.markdown(f"**Per-timepoint significance ({treatment_choice}, FDR q<0.05 across all 15 population x day tests)**")
            if len(significant_points):
                st.dataframe(significant_points.round(4), width="stretch", hide_index=True)
            else:
                st.info(f"No population diverges by response at any single timepoint under {treatment_choice}.")
            with st.expander("Full per-timepoint test results"):
                st.dataframe(arm_stats.round(4), width="stretch", hide_index=True)

# Part 4 baseline subset
with tab_subset:
    st.subheader("Baseline (t=0) melanoma PBMC samples treated with miraclib")

    if subset_breakdown is not None:
        col_a, col_b, col_c = st.columns(3)
        for col, breakdown_name, title in (
            (col_a, "samples_per_project", "Samples per project"),
            (col_b, "responders_by_subject", "Subjects by response"),
            (col_c, "sex_by_subject", "Subjects by sex"),
        ):
            rows = subset_breakdown[subset_breakdown["breakdown"] == breakdown_name]
            with col:
                st.markdown(f"**{title}**")
                for _, row in rows.iterrows():
                    st.metric(str(row["category"]), int(row["count"]))

    st.dataframe(baseline, width='stretch', hide_index=True)

    st.divider()
    st.subheader("Melanoma male responders, all samples/treatments, at t=0")
    st.caption("Average B-cell count - not restricted to PBMC or to miraclib.")
    if answer_path.exists():
        answer = answer_path.read_text(encoding="utf-8").strip().split(": ")[-1]
        st.metric("Average B-cell count", answer)

    if derived_features is not None:
        st.divider()
        st.subheader("Derived ratio features (for downstream modeling)")
        st.caption(
            "CD4:CD8, NK:monocyte, and lymphoid:myeloid ratios computed per PBMC "
            "miraclib melanoma sample - classic immunology-derived features that "
            "compress the five-population mix into a few interpretable numbers."
        )
        st.dataframe(derived_features.head(200), width='stretch', hide_index=True)

# Signal model, exploratory baseline for predicting response from cell mix
with tab_signal:
    st.subheader("Can the five-population mix predict miraclib response?")
    st.caption(
        "Population percentages are compositional (they sum to 100% per sample), so "
        "they're centered-log-ratio (CLR) transformed before use in correlation, PCA, "
        "and the logistic regression below - this avoids treating the shared "
        "sum-to-100 constraint as if it were real biological signal."
    )

    if signal_metrics is not None:
        m = signal_metrics.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Out-of-fold ROC-AUC", f"{m['oof_roc_auc']:.3f}")
        c2.metric("Permutation p-value", f"{m['auc_permutation_p_value']:.3f}")
        c3.metric("Subjects (grouped CV)", int(m["n_subjects"]))
        c4.metric("Responders / non-responders", f"{int(m['n_responders'])} / {int(m['n_non_responders'])}")

        n_responders, n_non_responders = int(m["n_responders"]), int(m["n_non_responders"])
        n_samples = n_responders + n_non_responders
        majority_baseline = max(n_responders, n_non_responders) / n_samples
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Out-of-fold accuracy", f"{m['oof_accuracy']:.1%}",
            delta=f"{(m['oof_accuracy'] - majority_baseline):+.1%} vs. always-guess-majority",
        )
        d2.metric("Always-guess-majority baseline", f"{majority_baseline:.1%}")
        d3.metric("Mean fold AUC", f"{m['mean_fold_auc']:.3f} ± {m['std_fold_auc']:.3f}")

        n_beating = int(m["n_null_beating_observed"]) if "n_null_beating_observed" in m else 0
        st.caption(
            f"Cross-validation: {m['cv_scheme']}. The AUC ({m['oof_roc_auc']:.3f}) is modest in "
            f"absolute terms, but a {int(m['n_permutations'])}-permutation label-shuffle test shows "
            f"only {n_beating} of {int(m['n_permutations'])} shuffled-label runs matched or beat it "
            f"(p={m['auc_permutation_p_value']:.4f}) - a real, "
            "reproducible signal, just too weak on these five features alone to be clinically useful. "
            "Accuracy barely clears the always-guess-majority baseline, which is expected at this AUC: "
            "a model needs to separate classes by a wide margin before accuracy (a 0.5-threshold call) "
            "moves much past just picking the larger class every time - AUC is the more honest number "
            "to report here."
        )

    col_perm, col_folds = st.columns(2)
    with col_perm:
        if signal_permutation_null is not None and signal_metrics is not None:
            fig_perm = px.histogram(
                signal_permutation_null, x="null_auc", nbins=30,
                labels={"null_auc": "AUC under shuffled labels"},
                title="Observed AUC vs. the label-shuffled null distribution",
            )
            fig_perm.update_traces(marker_color="#898781")
            fig_perm.add_vline(
                x=signal_metrics.iloc[0]["oof_roc_auc"], line_color="#2a78d6", line_width=2,
                annotation_text="Observed AUC", annotation_position="top",
            )
            st.plotly_chart(fig_perm, width="stretch")

    with col_folds:
        if signal_cv_folds is not None and signal_metrics is not None:
            fig_folds = px.bar(
                signal_cv_folds, x="cv_fold", y="fold_auc",
                labels={"cv_fold": "Fold", "fold_auc": "AUC"},
                title="Out-of-fold AUC by cross-validation fold",
            )
            fig_folds.update_traces(marker_color="#2a78d6")
            fig_folds.add_hline(
                y=signal_metrics.iloc[0]["oof_roc_auc"], line_color="#898781", line_dash="dash",
                annotation_text="Pooled OOF AUC", annotation_position="bottom right",
            )
            fig_folds.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_folds, width="stretch")
            st.caption(
                "Per-fold AUC varies with which subjects land in the held-out fold - this "
                "spread is why the pooled out-of-fold AUC (dashed line) is the number to "
                "trust over any single fold."
            )

    col_roc, col_coef = st.columns(2)
    with col_roc:
        if signal_roc is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=signal_roc["fpr"], y=signal_roc["tpr"], mode="lines",
                           name="Model", line=dict(color="#2a78d6", width=2))
            )
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                           line=dict(color="#898781", width=1, dash="dash"))
            )
            fig_roc.update_layout(
                title="Out-of-fold ROC curve",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_roc, width='stretch')

    with col_coef:
        if signal_coefficients is not None:
            coef_sorted = signal_coefficients.sort_values("standardized_coefficient")
            fig_coef = px.bar(
                coef_sorted,
                x="standardized_coefficient",
                y="population",
                orientation="h",
                color="standardized_coefficient",
                color_continuous_scale=DIVERGING_SCALE,
                color_continuous_midpoint=0,
                labels={"standardized_coefficient": "Standardized coefficient (log-odds)"},
                title="Logistic regression coefficients (CLR features)",
            )
            fig_coef.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_coef, width='stretch')
            st.caption(
                "Odds ratio = e^(coefficient): how much the odds of response multiply "
                "per 1-SD increase in that population's CLR-transformed share, holding "
                "the other four fixed."
            )
            st.dataframe(
                coef_sorted[["population", "standardized_coefficient", "odds_ratio"]]
                .sort_values("standardized_coefficient", key=lambda s: s.abs(), ascending=False)
                .round(3),
                width='stretch', hide_index=True,
            )

    col_corr, col_pca = st.columns(2)
    with col_corr:
        if correlations is not None:
            corr_indexed = correlations.set_index(correlations.columns[0])[POPULATIONS].loc[POPULATIONS]
            fig_corr = px.imshow(
                corr_indexed,
                color_continuous_scale=DIVERGING_SCALE,
                zmin=-1, zmax=1,
                text_auto=".2f",
                labels={"color": "Spearman corr."},
                title="Population correlation (Spearman)",
            )
            st.plotly_chart(fig_corr, width='stretch')

    with col_pca:
        if signal_pca is not None:
            variance = ""
            if signal_pca_variance is not None:
                v1, v2 = signal_pca_variance["explained_variance_ratio"]
                variance = f" (PC1 {v1:.0%}, PC2 {v2:.0%} of variance)"
            fig_pca = px.scatter(
                signal_pca,
                x="pc1", y="pc2", color="response",
                color_discrete_map=RESPONSE_COLORS,
                opacity=0.6,
                labels={"pc1": "PC1", "pc2": "PC2", "response": "Response"},
                title=f"PCA of CLR-transformed population mix{variance}",
            )
            st.plotly_chart(fig_pca, width='stretch')

    col_loadings, col_calibration = st.columns(2)
    with col_loadings:
        if signal_pca_loadings is not None:
            loadings_long = signal_pca_loadings.melt(
                id_vars="population", value_vars=["pc1_loading", "pc2_loading"],
                var_name="component", value_name="loading",
            )
            loadings_long["component"] = loadings_long["component"].map(
                {"pc1_loading": "PC1", "pc2_loading": "PC2"}
            )
            fig_load = px.bar(
                loadings_long, x="population", y="loading", color="component",
                barmode="group",
                category_orders={"population": POPULATIONS},
                color_discrete_map={"PC1": "#2a78d6", "PC2": "#eb6834"},
                labels={"loading": "Loading", "population": "Population", "component": "Component"},
                title="What drives PC1 vs. PC2",
            )
            st.plotly_chart(fig_load, width='stretch')
            st.caption(
                "Each bar is that population's weight in the PCA axis above - the "
                "larger the magnitude, the more that population's CLR value moves a "
                "sample along that axis. b_cell dominates PC1; nk_cell dominates PC2."
            )

    with col_calibration:
        if signal_oof_predictions is not None:
            fig_calib = px.histogram(
                signal_oof_predictions, x="oof_probability", color="response",
                barmode="overlay", opacity=0.65, nbins=30,
                color_discrete_map=RESPONSE_COLORS,
                labels={"oof_probability": "Predicted probability of response", "response": "Actual response"},
                title="Out-of-fold predicted probability, by actual response",
            )
            fig_calib.add_vline(x=0.5, line_color="#898781", line_dash="dash", annotation_text="0.5 cutoff")
            st.plotly_chart(fig_calib, width='stretch')
            st.caption(
                "If the model separated the classes well, the two colors would sit in "
                "different ranges either side of 0.5. Instead they largely overlap - the "
                "visual counterpart of the modest AUC: the model shifts the odds a little, "
                "it doesn't sort responders from non-responders."
            )
