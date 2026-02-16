"""
Type-specific regressions of ECI_software on pre-treatment predictors.
Pooled OLS baseline, per-type OLS with HC1, Chow test, forest plot.
Specification: eci_software ~ ln_pob + sec_educ + pc + nbi + radiance.
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import text
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

engine = sqlalchemy.create_engine("postgresql://postgres:postgres@localhost/posadas")
df = pd.read_sql("SELECT * FROM art1.departamentos", engine)
print(f"\nLoaded {len(df)} departments, {df.shape[1]} columns")

# Verify cluster assignments
if "mca_cluster" not in df.columns or df["mca_cluster"].isna().all():
    raise RuntimeError("No cluster assignments found. Run analysis_51_cah.py first.")

df["ln_pob_2010"] = np.log(df["pob_2010"].replace(0, np.nan))
n_clusters = df["mca_cluster"].dropna().nunique()
print(f"  Clusters found: {n_clusters}")
print(f"  Cluster distribution:")
for c in sorted(df["mca_cluster"].dropna().unique()):
    label = df.loc[df["mca_cluster"] == c, "mca_cluster_label"].iloc[0] \
        if "mca_cluster_label" in df.columns else f"Cluster {int(c)}"
    n = (df["mca_cluster"] == c).sum()
    n_eci = df.loc[df["mca_cluster"] == c, "eci_software"].notna().sum()
    print(f"    {label}: N={n} ({n_eci} with ECI)")

# ---
# Model specification
# ---

predictors = [
    "ln_pob_2010",
    "pct_jefe_sec_2010",
    "pct_pc_2010",
    "pct_nbi_2010",
    "rad_2014",
]

dv = "eci_software"

var_labels = {
    "ln_pob_2010": "ln(Population)",
    "pct_jefe_sec_2010": "% Sec. educ. HH heads",
    "pct_pc_2010": "% Computer ownership",
    "pct_nbi_2010": "% Structural poverty (NBI)",
    "rad_2014": "Nighttime radiance",
    "const": "Constant",
}

print(f"  DV: {dv}")
print(f"  Predictors: {', '.join(predictors)}")
print(f"  No region dummies (types capture spatial structure)")


# ---
# Helpers
# ---

def sig_star(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    elif p < 0.10: return "."
    else: return ""


def run_ols(data, predictors, dv, title):
    """Run OLS with HC1 errors, print results, return model."""
    cols = predictors + [dv]
    clean = data.dropna(subset=cols)
    n = len(clean)

    if n < len(predictors) + 2:
        print(f"\n  {title}: N = {n} — TOO FEW OBSERVATIONS FOR OLS")
        return None, clean

    X = add_constant(clean[predictors])
    y = clean[dv]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print(f"\n  {'~' * 70}")
    print(f"  {title}")
    print(f"  {'~' * 70}")
    print(f"  N = {int(model.nobs)}, R² = {model.rsquared:.4f}, "
          f"Adj R² = {model.rsquared_adj:.4f}")
    print(f"  F = {model.fvalue:.3f}, p(F) = {model.f_pvalue:.2e}")

    print(f"\n  {'Variable':<30} {'B':>8} {'SE(HC1)':>10} {'t':>8} {'p':>10} {'Beta':>8}")
    print(f"  {'-'*74}")

    # Standardised coefficients
    X_std = clean[predictors].copy()
    y_std = (y - y.mean()) / y.std()
    for c in predictors:
        X_std[c] = (X_std[c] - X_std[c].mean()) / X_std[c].std()
    X_std = add_constant(X_std)
    model_std = sm.OLS(y_std, X_std).fit(cov_type="HC1")

    for var in model.params.index:
        b = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        beta = model_std.params.get(var, np.nan)
        label = var_labels.get(var, var)
        s = sig_star(p)
        if var == "const":
            print(f"  {label:<30} {b:>8.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f} {'---':>8}")
        else:
            print(f"  {label:<30} {b:>8.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f} {beta:>8.3f} {s}")

    return model, clean


def compute_standardised_betas(data, predictors, dv):
    """Return dict of standardised betas and SEs."""
    cols = predictors + [dv]
    clean = data.dropna(subset=cols)
    if len(clean) < len(predictors) + 2:
        return None

    X = clean[predictors].copy()
    y = clean[dv].copy()
    y_std = (y - y.mean()) / y.std()
    for c in predictors:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = add_constant(X)
    model = sm.OLS(y_std, X).fit(cov_type="HC1")

    result = {}
    for var in predictors:
        result[var] = {
            "beta": model.params[var],
            "se": model.bse[var],
            "p": model.pvalues[var],
            "ci_low": model.conf_int().loc[var, 0],
            "ci_high": model.conf_int().loc[var, 1],
        }
    return result


def bootstrap_correlations(data, var1, var2, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for Pearson r."""
    clean = data[[var1, var2]].dropna()
    n = len(clean)
    if n < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(seed)
    r_obs = clean[var1].corr(clean[var2])
    boot_r = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = clean.iloc[idx]
        boot_r[b] = sample[var1].corr(sample[var2])

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_r, alpha * 100)
    ci_high = np.percentile(boot_r, (1 - alpha) * 100)
    return r_obs, ci_low, ci_high


# ---
# Pooled model
# ---

pooled_data = df[df["eci_software"].notna()].copy()
model_pooled, data_pooled = run_ols(pooled_data, predictors, dv,
                                     "POOLED OLS: eci_software (all types)")

# VIF for pooled
if model_pooled is not None:
    clean_pooled = data_pooled.dropna(subset=predictors + [dv])
    X_vif = add_constant(clean_pooled[predictors])
    print(f"\n  VIF:")
    for i, col in enumerate(X_vif.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X_vif.values, i)
        label = var_labels.get(col, col)
        flag = " << HIGH" if vif > 5 else (" << MODERATE" if vif > 2.5 else "")
        print(f"    {label:<30} VIF = {vif:.2f}{flag}")

# ---
# Regressions by type
# ---

type_results = {}
type_betas = {}

for c in sorted(df["mca_cluster"].dropna().unique()):
    c_int = int(c)
    label = df.loc[df["mca_cluster"] == c, "mca_cluster_label"].iloc[0] \
        if "mca_cluster_label" in df.columns else f"Cluster {c_int}"

    subset = df[df["mca_cluster"] == c].copy()
    n_eci = subset["eci_software"].notna().sum()

    if n_eci >= 30:
        model_c, data_c = run_ols(
            subset, predictors, dv,
            f"TYPE {c_int}: {label} (N_eci = {n_eci})"
        )
        type_results[c_int] = model_c
        betas = compute_standardised_betas(subset, predictors, dv)
        type_betas[c_int] = betas
    else:
        print(f"\n  TYPE {c_int}: {label} — N_eci = {n_eci} < 30")
        print(f"  Using bivariate correlations with bootstrap CI instead.")
        type_results[c_int] = None
        type_betas[c_int] = None

        eci_sub = subset[subset["eci_software"].notna()]
        print(f"\n  {'Predictor':<30} {'r':>8} {'CI low':>8} {'CI high':>8} {'N':>5}")
        print(f"  {'-'*59}")
        for var in predictors:
            r, ci_lo, ci_hi = bootstrap_correlations(eci_sub, var, dv)
            n_v = eci_sub[[var, dv]].dropna().shape[0]
            label_v = var_labels.get(var, var)
            print(f"  {label_v:<30} {r:>8.3f} {ci_lo:>8.3f} {ci_hi:>8.3f} {n_v:>5}")

# ---
# Chow test
# ---

# Pooled model with interactions: type × each predictor
eci_data = df[df["eci_software"].notna() & df["mca_cluster"].notna()].copy()
eci_data = eci_data.dropna(subset=predictors + [dv, "mca_cluster"])

# Create type dummies (reference = first cluster)
clusters_sorted = sorted(eci_data["mca_cluster"].unique())
ref_cluster = clusters_sorted[0]
type_dummies = pd.get_dummies(eci_data["mca_cluster"], prefix="type", dtype=float)
type_dummy_cols = [c for c in type_dummies.columns if c != f"type_{ref_cluster}"]
eci_data = pd.concat([eci_data, type_dummies[type_dummy_cols]], axis=1)

# Restricted model (no interactions)
X_restricted = add_constant(eci_data[predictors])
y = eci_data[dv]
model_restricted = sm.OLS(y, X_restricted).fit()
rss_r = model_restricted.ssr
k_r = model_restricted.df_model + 1  # +1 for constant
n_total = len(y)

# Unrestricted model (with type dummies + interactions)
interaction_cols = []
for dummy in type_dummy_cols:
    for pred in predictors:
        col_name = f"{dummy}_x_{pred}"
        eci_data[col_name] = eci_data[dummy] * eci_data[pred]
        interaction_cols.append(col_name)

X_unrestricted = add_constant(
    eci_data[predictors + type_dummy_cols + interaction_cols]
)
model_unrestricted = sm.OLS(y, X_unrestricted).fit()
rss_u = model_unrestricted.ssr
k_u = model_unrestricted.df_model + 1

# Chow F-statistic
q = k_u - k_r  # number of restrictions
f_chow = ((rss_r - rss_u) / q) / (rss_u / (n_total - k_u))
p_chow = 1 - stats.f.cdf(f_chow, q, n_total - k_u)

print(f"\n  Restricted (pooled): RSS = {rss_r:.4f}, k = {k_r}")
print(f"  Unrestricted (interactions): RSS = {rss_u:.4f}, k = {k_u}")
print(f"  Restrictions: q = {q}")
print(f"  Chow F = {f_chow:.4f}")
print(f"  p-value = {p_chow:.4f}")
if p_chow < 0.05:
    print(f"  -> SIGNIFICANT: coefficients differ across types (p < 0.05)")
else:
    print(f"  -> NOT significant at p < 0.05")

# Also report unrestricted R²
print(f"\n  Restricted R² = {model_restricted.rsquared:.4f}")
print(f"  Unrestricted R² = {model_unrestricted.rsquared:.4f}")
print(f"  R² gain from interactions = {model_unrestricted.rsquared - model_restricted.rsquared:.4f}")

# ---
# Coefficient comparison
# ---

# Pooled betas
pooled_betas = compute_standardised_betas(pooled_data, predictors, dv)

header = f"  {'Variable':<25} {'Pooled':>10}"
for c in clusters_sorted:
    c_int = int(c)
    header += f" {'Type '+str(c_int):>10}"
print(header)
print(f"  {'-'*25 + '-'*10 + '-'*10*len(clusters_sorted)}")

for var in predictors:
    label = var_labels.get(var, var)
    row = f"  {label:<25}"

    # Pooled
    if pooled_betas and var in pooled_betas:
        b = pooled_betas[var]["beta"]
        p = pooled_betas[var]["p"]
        row += f" {b:>7.3f}{sig_star(p):<3}"
    else:
        row += f" {'---':>10}"

    # Per type
    for c in clusters_sorted:
        c_int = int(c)
        betas = type_betas.get(c_int)
        if betas and var in betas:
            b = betas[var]["beta"]
            p = betas[var]["p"]
            row += f" {b:>7.3f}{sig_star(p):<3}"
        else:
            row += f" {'---':>10}"

    print(row)

# N and R² row
row_n = f"  {'N':<25}"
row_r2 = f"  {'R²':<25}"
if model_pooled:
    row_n += f" {int(model_pooled.nobs):>10}"
    row_r2 += f" {model_pooled.rsquared:>10.3f}"
for c in clusters_sorted:
    c_int = int(c)
    m = type_results.get(c_int)
    if m:
        row_n += f" {int(m.nobs):>10}"
        row_r2 += f" {m.rsquared:>10.3f}"
    else:
        row_n += f" {'<30':>10}"
        row_r2 += f" {'---':>10}"
print(row_n)
print(row_r2)

# ---
# Forest plot
# ---

# Collect all betas for plotting
plot_data = []

# Pooled
if pooled_betas:
    for var in predictors:
        if var in pooled_betas:
            plot_data.append({
                "variable": var_labels.get(var, var),
                "type": "Pooled",
                "beta": pooled_betas[var]["beta"],
                "ci_low": pooled_betas[var]["ci_low"],
                "ci_high": pooled_betas[var]["ci_high"],
                "p": pooled_betas[var]["p"],
            })

# Per type
for c in clusters_sorted:
    c_int = int(c)
    label_c = df.loc[df["mca_cluster"] == c, "mca_cluster_label"].iloc[0] \
        if "mca_cluster_label" in df.columns else f"Type {c_int}"
    betas = type_betas.get(c_int)
    if betas:
        for var in predictors:
            if var in betas:
                plot_data.append({
                    "variable": var_labels.get(var, var),
                    "type": label_c,
                    "beta": betas[var]["beta"],
                    "ci_low": betas[var]["ci_low"],
                    "ci_high": betas[var]["ci_high"],
                    "p": betas[var]["p"],
                })

plot_df = pd.DataFrame(plot_data)

if len(plot_df) > 0:
    variables = list(dict.fromkeys(plot_df["variable"]))
    types = list(dict.fromkeys(plot_df["type"]))
    n_vars = len(variables)
    n_types = len(types)

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, n_vars * 1.5)))

    colours = ["black", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    offsets = np.linspace(-0.3, 0.3, n_types)

    for j, type_name in enumerate(types):
        sub = plot_df[plot_df["type"] == type_name]
        for _, row in sub.iterrows():
            var_idx = variables.index(row["variable"])
            y_pos = n_vars - 1 - var_idx + offsets[j]
            colour = colours[j % len(colours)]
            marker = "D" if type_name == "Pooled" else "o"
            size = 8 if type_name == "Pooled" else 6

            ax.errorbar(
                row["beta"], y_pos,
                xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
                fmt=marker, color=colour, markersize=size, capsize=3, capthick=1,
                linewidth=1.5, alpha=0.85
            )

    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(list(reversed(variables)), fontsize=10)
    ax.set_xlabel("Standardised coefficient (Beta) with 95% CI", fontsize=11)
    ax.set_title("Forest Plot: Determinants of ECIsoftware by Departmental Type", fontsize=12)

    # Legend
    legend_handles = []
    for j, type_name in enumerate(types):
        colour = colours[j % len(colours)]
        marker = "D" if type_name == "Pooled" else "o"
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor=colour,
                       markersize=8, label=type_name)
        )
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_06_forest_plot.png"),
                dpi=300, bbox_inches="tight")
    print("  Forest plot saved: fig_forest_plot.png")
    plt.close()
else:
    print("  WARNING: No data for forest plot.")

# ---
# Cultural capital by type
# ---

print(f"\n  Standardised Beta of pct_jefe_sec_2010 by type:")
sec_var = "pct_jefe_sec_2010"
if pooled_betas and sec_var in pooled_betas:
    b = pooled_betas[sec_var]["beta"]
    p = pooled_betas[sec_var]["p"]
    print(f"    Pooled:          Beta = {b:.3f} (p = {p:.4f})")

for c in clusters_sorted:
    c_int = int(c)
    label_c = df.loc[df["mca_cluster"] == c, "mca_cluster_label"].iloc[0] \
        if "mca_cluster_label" in df.columns else f"Type {c_int}"
    betas = type_betas.get(c_int)
    if betas and sec_var in betas:
        b = betas[sec_var]["beta"]
        p = betas[sec_var]["p"]
        print(f"    {label_c:<20} Beta = {b:.3f} (p = {p:.4f})")
    else:
        print(f"    {label_c:<20} (N < 30, OLS not estimated)")

print(f"\n  Interpretation:")
print(f"    If Beta(sec. educ.) varies substantially across types, this confirms")
print(f"    that the effect of cultural capital is contingent on territorial context.")
print(f"    The pooled model masks this heterogeneity.")

