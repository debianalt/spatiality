"""
09_eci_temporal_panel.py

Compute ECI_software for cumulative snapshots 2015-2025.
Each snapshot uses all repos created up to December 31 of year t,
applying identical methodology to 01_compute_eci.py.

Outputs:
  data/eci_panel_2015_2025.csv         -- ECI by dept by year
  figures/fig_panel_trajectories.png   -- ECI trajectories by cluster type
  figures/fig_panel_rank_stability.png -- Spearman rho heatmap + 2015 vs 2025 scatter

Printed statistics:
  Spearman rho ECI_2015 vs ECI_2025
  Quintile persistence 2015->2025
  Top-quintile persistence
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from scipy.linalg import eig
from scipy.stats import spearmanr as _spearmanr

def spearmanr(a, b):
    """Wrapper that always returns (float rho, float pval)."""
    res = _spearmanr(np.asarray(a), np.asarray(b))
    # scipy >= 1.9 returns SpearmanrResult with .statistic/.pvalue (arrays if
    # multiple columns); older returns (rho, pval) scalars directly.
    try:
        rho  = float(np.ravel(res.statistic)[0])
        pval = float(np.ravel(res.pvalue)[0])
    except AttributeError:
        rho, pval = float(res[0]), float(res[1])
    return rho, pval
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_URL = "postgresql://postgres:postgres@localhost:5432/posadas"
engine = create_engine(ENGINE_URL)

YEARS = list(range(2015, 2026))

MIN_DEPT_REPOS = 10
MIN_LANG_REPOS = 30

FOREIGN_USERS = pd.read_csv(
    REPO_ROOT / "audit" / "audit_04_foreign_users.csv",
    usecols=["username"],
)["username"].tolist()
print(f"Foreign-user exclusion list: {len(FOREIGN_USERS)} users")

CORDOBA_CORRECTIONS = {
    "14112": "14119",
    "14119": "14126",
    "14126": "14133",
    "14133": "14140",
    "14140": "14147",
    "14154": "14161",
    "14175": "14182",
    "14182": "14112",
}

MAPPING = {
    "06217": "06218",
    "06466": "06218",
    "94007": "94008",
    "94014": "94015",
    "94011": "94015",
    **CORDOBA_CORRECTIONS,
}

EXCLUDE_CODES = {"94021", "94028"}

CABA_PREFIX = "02"


def apply_corrections(df):
    df = df.copy()
    df["dpto5"] = df["dpto5"].str.zfill(5)
    df["dpto5"] = df["dpto5"].apply(
        lambda x: "02000" if x.startswith(CABA_PREFIX) else MAPPING.get(x, x)
    )
    df = df[~df["dpto5"].isin(EXCLUDE_CODES)].copy()
    return df


def compute_eci_year(year):
    cutoff = f"{year}-12-31T23:59:59Z"
    query = text("""
        SELECT LEFT(redcode, 5) AS dpto5,
               primary_language,
               COUNT(*)         AS repos
        FROM   github_argentina.repos
        WHERE  primary_language IS NOT NULL
          AND  primary_language != ''
          AND  username NOT IN (SELECT unnest(:fu))
          AND  created_at <= :cutoff
        GROUP  BY 1, 2
    """)
    df = pd.read_sql(query, engine, params={"fu": FOREIGN_USERS, "cutoff": cutoff})
    df = apply_corrections(df)
    df = df.groupby(["dpto5", "primary_language"], as_index=False)["repos"].sum()

    # Thresholds
    lang_totals = df.groupby("primary_language")["repos"].sum()
    valid_langs = lang_totals[lang_totals >= MIN_LANG_REPOS].index
    dept_totals = df.groupby("dpto5")["repos"].sum()
    valid_depts = dept_totals[dept_totals >= MIN_DEPT_REPOS].index

    df = df[df["primary_language"].isin(valid_langs) & df["dpto5"].isin(valid_depts)].copy()

    n_d = df["dpto5"].nunique()
    n_l = df["primary_language"].nunique()
    if n_d < 20 or n_l < 5:
        print(f"  {year}: skipped (n_depts={n_d}, n_langs={n_l})")
        return None

    # Bipartite matrix
    M = df.pivot_table(index="dpto5", columns="primary_language",
                       values="repos", fill_value=0)
    M_arr = M.values.astype(float)

    # RCA -> binary A
    row_s = M_arr.sum(axis=1, keepdims=True)
    col_s = M_arr.sum(axis=0, keepdims=True)
    total = M_arr.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        RCA = (M_arr / row_s) / (col_s / total)
    RCA = np.nan_to_num(RCA, nan=0.0)
    A = (RCA >= 1).astype(float)

    diversity = A.sum(axis=1)
    ubiquity  = A.sum(axis=0)
    diversity[diversity == 0] = 1e-10
    ubiquity[ubiquity == 0]   = 1e-10

    D_inv = np.diag(1.0 / diversity)
    U_inv = np.diag(1.0 / ubiquity)

    M_tilde = D_inv @ A @ U_inv @ A.T
    eigenvalues, eigenvectors = eig(M_tilde)
    idx = np.argsort(-eigenvalues.real)
    eci_raw = eigenvectors[:, idx[1]].real
    eci_std = (eci_raw - eci_raw.mean()) / eci_raw.std()

    # Sign: high ECI should correlate positively with diversity
    if np.corrcoef(eci_std, diversity)[0, 1] < 0:
        eci_std = -eci_std

    avg_ubiq = np.array([
        ubiquity[A[d, :] == 1].mean() if A[d, :].any() else np.nan
        for d in range(n_d)
    ])

    result = pd.DataFrame({
        "dpto5":         M.index,
        "year":          year,
        "eci_software":  eci_std,
        "n_languages":   (A > 0).sum(axis=1),
        "n_repos":       M_arr.sum(axis=1),
        "avg_ubiquity":  avg_ubiq,
    })
    print(f"  {year}: {n_d} depts, {n_l} languages, "
          f"total repos={int(M_arr.sum()):,}")
    return result


# ---- Main loop ----
print("Computing annual ECI snapshots (cumulative)...")
records = []
for yr in YEARS:
    res = compute_eci_year(yr)
    if res is not None:
        records.append(res)

panel = pd.concat(records, ignore_index=True)
out_csv = REPO_ROOT / "data" / "eci_panel_2015_2025.csv"
panel.to_csv(out_csv, index=False)
print(f"\nPanel saved: {len(panel)} rows -> {out_csv}")

# ---- Merge cluster labels from art1 ----
baseline = pd.read_sql(
    "SELECT dpto5, mca_cluster_label FROM art1.departamentos",
    engine,
)
panel = panel.merge(baseline, on="dpto5", how="left")

# ---- Rank stability statistics ----
pivot = panel.pivot(index="dpto5", columns="year", values="eci_software")
years_avail = sorted(pivot.columns.tolist())

# 1. Consecutive-year rank stability (n varies each pair; most comparable)
print("\n--- Consecutive-year rank stability (Spearman rho, t vs t+1) ---")
for i in range(len(years_avail) - 1):
    y1, y2 = years_avail[i], years_avail[i + 1]
    common = pivot[[y1, y2]].dropna()
    if len(common) >= 10:
        rho, pval = spearmanr(common[y1], common[y2])
        print(f"  ECI_{y1} vs ECI_{y2}: rho={rho:.3f}, p={pval:.4f}, n={len(common)}")

# 2. Base-2020 stability: departments present in 2020 (N=125), tracked forward
print("\n--- Base-2020 stability (Spearman rho, departments present in 2020+) ---")
base_yr = 2020
for yr in [y for y in years_avail if y >= 2020]:
    common = pivot[[base_yr, yr]].dropna()
    if len(common) >= 10:
        rho, pval = spearmanr(common[base_yr], common[yr])
        print(f"  ECI_{base_yr} vs ECI_{yr}: rho={rho:.3f}, p={pval:.4f}, n={len(common)}")

# 3. Fixed-cohort analysis: departments present in all years 2018-2025
years_stable = [y for y in years_avail if y >= 2018]
stable_depts = pivot[years_stable].dropna(how="any").index
print(f"\n--- Fixed-cohort (depts present 2018-2025, N={len(stable_depts)}) ---")
pivot_stable = pivot.loc[stable_depts, years_stable]
for yr in years_stable:
    common = pivot_stable[[years_stable[0], yr]].dropna()
    if len(common) >= 10:
        rho, pval = spearmanr(common[years_stable[0]], common[yr])
        print(f"  ECI_{years_stable[0]} vs ECI_{yr}: rho={rho:.3f}, p={pval:.4f}, n={len(common)}")

# 4. Quintile persistence: fixed cohort 2020-2025
if 2020 in pivot.columns and 2025 in pivot.columns:
    common = pivot[[2020, 2025]].dropna()
    q20 = pd.qcut(common[2020], q=5, labels=False) + 1
    q25 = pd.qcut(common[2025], q=5, labels=False) + 1
    same_q = float((q20 == q25).mean())
    top_persist = float(((q20 == 5) & (q25 == 5)).sum() / (q20 == 5).sum())
    rho_20_25, pval_20_25 = spearmanr(common[2020], common[2025])
    print(f"\nQuintile persistence 2020->2025: {same_q:.1%} remain in same quintile (n={len(common)})")
    print(f"Top-quintile persistence 2020->2025: {top_persist:.1%}")
    print(f"ECI_2020 vs ECI_2025: Spearman rho={rho_20_25:.3f} (p={pval_20_25:.4f}, n={len(common)})")

# ---- FIGURE A: Trajectories by cluster type ----
CLUSTER_ORDER = [
    "Metropolitan-Core",
    "Metropolitan-Diversified",
    "Pampeana-Educated",
    "Intermediate-Urban",
    "Peripheral-Deprived",
    "Semi-Rural-Active",
]
CLUSTER_COLORS = {
    "Metropolitan-Core":       "#b03a2e",
    "Metropolitan-Diversified":"#2874a6",
    "Pampeana-Educated":       "#1e8449",
    "Intermediate-Urban":      "#7d3c98",
    "Peripheral-Deprived":     "#ca6f1e",
    "Semi-Rural-Active":       "#5b2c6f",
}
CLUSTER_SHORT = {
    "Metropolitan-Core":       "Metro-Core",
    "Metropolitan-Diversified":"Metro-Div.",
    "Pampeana-Educated":       "Pamp.-Educ.",
    "Intermediate-Urban":      "Interm.-Urban",
    "Peripheral-Deprived":     "Periph.-Deprived",
    "Semi-Rural-Active":       "Semi-Rural-Act.",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: trajectories
ax = axes[0]
for ctype in CLUSTER_ORDER:
    grp = panel[panel["mca_cluster_label"] == ctype].groupby("year")["eci_software"]
    mean_eci = grp.mean()
    se_eci   = grp.sem()
    ax.plot(mean_eci.index, mean_eci.values,
            color=CLUSTER_COLORS[ctype], linewidth=2,
            marker="o", markersize=4,
            label=CLUSTER_SHORT[ctype])
    ax.fill_between(mean_eci.index,
                    mean_eci - se_eci,
                    mean_eci + se_eci,
                    color=CLUSTER_COLORS[ctype], alpha=0.12)

ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Mean ECI$_{software}$ (standardised)", fontsize=10)
ax.set_title("(a) ECI trajectories by departmental type", fontsize=11, loc="left")
ax.legend(fontsize=8, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

# Panel B: ECI_2020 vs ECI_2025 scatter (n=125, rho~0.78)
ax2 = axes[1]
if 2020 in pivot.columns and 2025 in pivot.columns:
    common_df = pivot[[2020, 2025]].dropna().reset_index()
    common_df = common_df.merge(baseline, on="dpto5", how="left")
    for ctype in CLUSTER_ORDER:
        sub = common_df[common_df["mca_cluster_label"] == ctype]
        ax2.scatter(sub[2020], sub[2025],
                    color=CLUSTER_COLORS[ctype], alpha=0.65, s=22,
                    label=CLUSTER_SHORT[ctype], zorder=3)

    lims = [
        min(common_df[2020].min(), common_df[2025].min()) - 0.1,
        max(common_df[2020].max(), common_df[2025].max()) + 0.1,
    ]
    ax2.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    rho_sc, _ = spearmanr(common_df[2020], common_df[2025])
    ax2.set_xlabel("ECI$_{software}$ 2020", fontsize=10)
    ax2.set_ylabel("ECI$_{software}$ 2025", fontsize=10)
    ax2.set_title(f"(b) ECI 2020 vs 2025 (Spearman rho = {rho_sc:.2f}, N = {len(common_df)})",
                  fontsize=11, loc="left")
    ax2.legend(fontsize=7.5, frameon=False, ncol=2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = REPO_ROOT / "figures" / "fig_panel_trajectories.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {fig_path}")

# ---- FIGURE B: Year-to-year Spearman rho matrix ----
n_yrs = len(years_avail)
rho_mat = np.full((n_yrs, n_yrs), np.nan)
for i, y1 in enumerate(years_avail):
    for j, y2 in enumerate(years_avail):
        common = pivot[[y1, y2]].dropna()
        if len(common) >= 10:
            rho_mat[i, j], _ = spearmanr(common[y1], common[y2])

fig2, ax3 = plt.subplots(figsize=(7, 6))
im = ax3.imshow(rho_mat, vmin=0.5, vmax=1.0, cmap="YlOrRd", aspect="auto")
ax3.set_xticks(range(n_yrs))
ax3.set_xticklabels(years_avail, rotation=45, ha="right", fontsize=9)
ax3.set_yticks(range(n_yrs))
ax3.set_yticklabels(years_avail, fontsize=9)
ax3.set_title("ECI rank stability: Spearman rho between snapshots", fontsize=11)
plt.colorbar(im, ax=ax3, label="Spearman rho")
for i in range(n_yrs):
    for j in range(n_yrs):
        if not np.isnan(rho_mat[i, j]):
            ax3.text(j, i, f"{rho_mat[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="black" if rho_mat[i, j] < 0.85 else "white")
plt.tight_layout()
fig2_path = REPO_ROOT / "figures" / "fig_panel_rank_stability.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved: {fig2_path}")

print("\nDone.")
