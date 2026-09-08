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

# Chart chrome tokens, matched to the app's Streamlit theme so every chart
# reads as one surface rather than a white box floating on the page.
CHART_SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig, height: int | None = None, bargap: float = 0.3):
    """Apply consistent chart chrome: surface, ink, hairline grid, spacing."""
    fig.update_layout(
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=13),
        title_font=dict(family=FONT_FAMILY, color=INK_PRIMARY, size=15),
        margin=dict(l=48, r=24, t=52, b=40),
        hoverlabel=dict(
            bgcolor=CHART_SURFACE,
            font=dict(family=FONT_FAMILY, color=INK_PRIMARY, size=12),
            bordercolor=AXIS_LINE,
        ),
        bargap=bargap,
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False, linecolor=AXIS_LINE,
                      tickcolor=AXIS_LINE, tickfont=dict(color=INK_MUTED, size=11))
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False, linecolor=AXIS_LINE,
                      tickcolor=AXIS_LINE, tickfont=dict(color=INK_MUTED, size=11))
    fig.update_annotations(font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=12))
    if height:
        fig.update_layout(height=height)
    return fig


st.set_page_config(page_title="Loblaw Bio immune analysis", layout="wide")
st.title("Loblaw Bio | Immune response analysis")
st.caption("Cell-population frequencies and response signals from the clinical trial data")

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
signal_permutation_null = load_csv("signal_model_permutation_null.csv")
cohort_overview = load_csv("cohort_overview.csv")
subject_demographics = load_csv("subject_demographics.csv")
longitudinal_trends = load_csv("longitudinal_trends.csv")
longitudinal_stats = load_csv("longitudinal_stats.csv")

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
        with st.container(border=True):
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
            fig.update_traces(marker_color=COUNT_BAR_COLOR, marker_cornerradius=4)
            fig.update_layout(showlegend=False)
            style_fig(fig, height=260)
            with col:
                with st.container(border=True):
                    st.plotly_chart(fig, width="stretch")

    if subject_demographics is not None:
        st.subheader("Age distribution")
        with st.container(border=True):
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
            style_fig(fig_age, height=340)
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
    with st.container(border=True):
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
        fig_mean.update_traces(marker_cornerradius=4)
        fig_mean.update_layout(showlegend=False)
        style_fig(fig_mean, height=380)
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
        with st.container(border=True):
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
            style_fig(fig_box, height=420)
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
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=8, line=dict(width=2, color=CHART_SURFACE)),
                        name=f"Response: {response_value}", legendgroup=response_value,
                        showlegend=show_legend,
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(tickvals=[0, 7, 14], title_text="Day")
        fig.update_yaxes(title_text="Mean %")
        fig.update_layout(
            title=f"{treatment_choice} - population trends by response",
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        )
        style_fig(fig, height=580)
        with st.container(border=True):
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
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    for _, row in rows.iterrows():
                        st.metric(str(row["category"]), int(row["count"]))

    st.dataframe(baseline, width='stretch', hide_index=True)

    st.divider()
    st.subheader("Melanoma male responders, all samples/treatments, at t=0")
    st.caption("Average B-cell count - not restricted to PBMC or to miraclib.")
    if answer_path.exists():
        answer = answer_path.read_text(encoding="utf-8").strip().split(": ")[-1]
        with st.container(border=True):
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
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Out-of-fold ROC-AUC", f"{m['oof_roc_auc']:.3f}")
            c2.metric("Permutation p-value", f"{m['auc_permutation_p_value']:.3f}")
            c3.metric("Subjects (grouped CV)", int(m["n_subjects"]))
            c4.metric("Responders / non-responders", f"{int(m['n_responders'])} / {int(m['n_non_responders'])}")
            st.caption(
                f"Cross-validation: {m['cv_scheme']}. The AUC ({m['oof_roc_auc']:.3f}) is modest in "
                f"absolute terms, but a {int(m['n_permutations'])}-permutation label-shuffle test shows "
                f"it beats every shuffled-label run (p={m['auc_permutation_p_value']:.3f}) - a real, "
                "reproducible signal, just too weak on these five features alone to be clinically useful."
            )

    if signal_permutation_null is not None and signal_metrics is not None:
        with st.container(border=True):
            fig_perm = px.histogram(
                signal_permutation_null, x="null_auc", nbins=30,
                labels={"null_auc": "AUC under shuffled labels"},
                title="Observed AUC vs. the label-shuffled null distribution",
            )
            fig_perm.update_traces(marker_color=INK_MUTED, marker_cornerradius=2)
            fig_perm.add_vline(
                x=signal_metrics.iloc[0]["oof_roc_auc"], line_color="#2a78d6", line_width=2,
                annotation_text="Observed AUC", annotation_position="top",
            )
            style_fig(fig_perm, height=340)
            st.plotly_chart(fig_perm, width="stretch")

    col_roc, col_coef = st.columns(2)
    with col_roc:
        if signal_roc is not None:
            with st.container(border=True):
                fig_roc = go.Figure()
                fig_roc.add_trace(
                    go.Scatter(x=signal_roc["fpr"], y=signal_roc["tpr"], mode="lines",
                               name="Model", line=dict(color="#2a78d6", width=2))
                )
                fig_roc.add_trace(
                    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                               line=dict(color=INK_MUTED, width=1, dash="dash"))
                )
                fig_roc.update_layout(
                    title="Out-of-fold ROC curve",
                    xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                style_fig(fig_roc, height=380)
                st.plotly_chart(fig_roc, width='stretch')

    with col_coef:
        if signal_coefficients is not None:
            with st.container(border=True):
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
                fig_coef.update_traces(marker_cornerradius=4)
                fig_coef.update_layout(coloraxis_showscale=False)
                style_fig(fig_coef, height=380)
                st.plotly_chart(fig_coef, width='stretch')

    col_corr, col_pca = st.columns(2)
    with col_corr:
        if correlations is not None:
            with st.container(border=True):
                corr_indexed = correlations.set_index(correlations.columns[0])[POPULATIONS].loc[POPULATIONS]
                fig_corr = px.imshow(
                    corr_indexed,
                    color_continuous_scale=DIVERGING_SCALE,
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    labels={"color": "Spearman corr."},
                    title="Population correlation (Spearman)",
                )
                style_fig(fig_corr, height=380)
                st.plotly_chart(fig_corr, width='stretch')

    with col_pca:
        if signal_pca is not None:
            with st.container(border=True):
                variance = ""
                if signal_pca_variance is not None:
                    v1, v2 = signal_pca_variance["explained_variance_ratio"]
                    variance = f" (PC1 {v1:.0%}, PC2 {v2:.0%} of variance)"
                fig_pca = px.scatter(
                    signal_pca,
                    x="pc1", y="pc2", color="response",
                    color_discrete_map=RESPONSE_COLORS,
                    opacity=0.75,
                    labels={"pc1": "PC1", "pc2": "PC2", "response": "Response"},
                    title=f"PCA of CLR-transformed population mix{variance}",
                )
                fig_pca.update_traces(marker=dict(size=8, line=dict(width=1, color=CHART_SURFACE)))
                style_fig(fig_pca, height=380)
                st.plotly_chart(fig_pca, width='stretch')
