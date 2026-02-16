"""
Ward's hierarchical clustering (CAH) on MCA factorial coordinates.
Input: Benzecri-retained axes from art1.departamentos.
Selects k via silhouette + Calinski-Harabasz, profiles clusters, saves assignments.
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import text
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---

print("Running CAH...")

engine = sqlalchemy.create_engine("postgresql://postgres:postgres@localhost/posadas")
df = pd.read_sql("SELECT * FROM art1.departamentos", engine)
print(f"\nLoaded {len(df)} departments, {df.shape[1]} columns")

# Identify MCA columns
mca_cols = [c for c in df.columns if c.startswith("mca_dim")]
print(f"  MCA dimensions available: {mca_cols}")

if not mca_cols:
    raise RuntimeError("No MCA dimensions found. Run analysis_50_mca.py first.")

# Filter to departments with MCA coordinates
df_mca = df.dropna(subset=mca_cols).copy()
print(f"  Departments with MCA coordinates: {len(df_mca)}")

X = df_mca[mca_cols].values
print(f"  Input matrix: {X.shape[0]} × {X.shape[1]}")

# ---
# Ward's linkage
# ---

print("\nWard's linkage...")

# Compute condensed distance matrix
dist_condensed = pdist(X, metric="euclidean")
print(f"  Pairwise distances: {len(dist_condensed)} pairs")
print(f"  Min dist: {dist_condensed.min():.4f}, Max: {dist_condensed.max():.4f}, "
      f"Mean: {dist_condensed.mean():.4f}")

# Ward's linkage
Z = linkage(dist_condensed, method="ward")
print(f"  Linkage matrix: {Z.shape}")

# ---
# Optimal k selection
# ---

print("\nOptimal k selection...")

k_range = range(3, 8)
silhouette_scores = {}
ch_scores = {}
cluster_sizes = {}

for k in k_range:
    labels = fcluster(Z, t=k, criterion="maxclust")
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    sizes = pd.Series(labels).value_counts().sort_index()
    min_size = sizes.min()

    silhouette_scores[k] = sil
    ch_scores[k] = ch
    cluster_sizes[k] = sizes

    print(f"  k={k}: silhouette={sil:.4f}, CH={ch:.1f}, "
          f"sizes={dict(sizes)}, min_size={min_size}")

# Best k by silhouette
best_k_sil = max(silhouette_scores, key=silhouette_scores.get)
# Also check that all clusters have N >= 40
for k in sorted(silhouette_scores, key=silhouette_scores.get, reverse=True):
    if cluster_sizes[k].min() >= 40:
        best_k = k
        break
else:
    best_k = best_k_sil
    print(f"  WARNING: No k satisfies min cluster size >= 40; using best silhouette k={best_k}")

print(f"\n  Selected k = {best_k} (silhouette = {silhouette_scores[best_k]:.4f})")

# Final cluster assignment
labels_final = fcluster(Z, t=best_k, criterion="maxclust")
df_mca["mca_cluster"] = labels_final

# ---
# Dendrogram
# ---

print("\nDendrogram...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=Z[-(best_k - 1), 2],
    ax=ax,
    above_threshold_color="grey"
)
ax.set_title(f"Ward's Dendrogram (N = {len(df_mca)}, cut at k = {best_k})", fontsize=13)
ax.set_xlabel("Cluster size")
ax.set_ylabel("Ward distance")
ax.axhline(Z[-(best_k - 1), 2], color="red", linestyle="--", linewidth=1.5,
           label=f"Cut at k = {best_k}")
ax.legend()
plt.tight_layout()
fig.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_S1_dendrogram.png"),
            dpi=300, bbox_inches="tight")
print("  Dendrogram saved: fig_cah_dendrogram.png")
plt.close()

# ---
# Silhouette analysis
# ---

print("\nSilhouette analysis...")

# Silhouette comparison across k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: silhouette by k
ks = list(k_range)
sils = [silhouette_scores[k] for k in ks]
chs = [ch_scores[k] for k in ks]
ax1.plot(ks, sils, "bo-", markersize=8, linewidth=2, label="Silhouette")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette score", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1b = ax1.twinx()
ax1b.plot(ks, chs, "rs--", markersize=8, linewidth=2, label="Calinski-Harabasz")
ax1b.set_ylabel("Calinski-Harabasz index", color="red")
ax1b.tick_params(axis="y", labelcolor="red")
ax1.set_title("Cluster quality metrics")
ax1.axvline(best_k, color="green", linestyle=":", linewidth=2, label=f"Selected k = {best_k}")
ax1.legend(loc="upper left")

# Right: silhouette per sample for selected k
sample_sil = silhouette_samples(X, labels_final)
y_lower = 10
for i in range(1, best_k + 1):
    cluster_sil = np.sort(sample_sil[labels_final == i])
    size_i = len(cluster_sil)
    y_upper = y_lower + size_i
    colour = cm.nipy_spectral(float(i) / (best_k + 1))
    ax2.fill_betweenx(
        np.arange(y_lower, y_upper), 0, cluster_sil,
        facecolor=colour, edgecolor=colour, alpha=0.7
    )
    ax2.text(-0.05, y_lower + 0.5 * size_i, f"C{i}", fontsize=10, fontweight="bold")
    y_lower = y_upper + 10

ax2.axvline(silhouette_scores[best_k], color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Silhouette coefficient")
ax2.set_ylabel("Departments (sorted)")
ax2.set_title(f"Silhouette plot (k = {best_k})")

plt.tight_layout()
fig.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_S2_cah_silhouette.png"),
            dpi=300, bbox_inches="tight")
print("  Silhouette plot saved: fig_cah_silhouette.png")
plt.close()

# Per-cluster silhouette means
print(f"\n  Per-cluster silhouette means (k = {best_k}):")
for i in range(1, best_k + 1):
    mask = labels_final == i
    mean_sil = sample_sil[mask].mean()
    n_i = mask.sum()
    print(f"    Cluster {i}: mean silhouette = {mean_sil:.4f}, N = {n_i}")

# ---
# Cluster profiles
# ---

print("\nCluster profiles...")

# Define profiling variables
profile_vars = [
    ("pob_2010", "Population 2010"),
    ("ln_pob_2010", "ln(Population) 2010"),
    ("pct_jefe_sec_2010", "% Sec. educ. HH heads 2010"),
    ("pct_jefe_uni_2010", "% Univ. educ. HH heads 2010"),
    ("pct_pc_2010", "% Computer ownership 2010"),
    ("pct_nbi_2010", "% Structural poverty (NBI) 2010"),
    ("pct_hacinam_2010", "% Overcrowding 2010"),
    ("rad_2014", "Nighttime radiance 2014"),
    ("tasa_empleo_2010", "Employment rate 2010"),
    ("gh_total_developers", "Total developers"),
    ("gh_devs_per_10k", "Developers per 10k"),
    ("gh_hill_q1_shannon", "Shannon diversity (Hill q=1)"),
    ("eci_software", "ECIsoftware"),
]

# Add ln_pob_2010 if not already present
if "ln_pob_2010" not in df_mca.columns:
    df_mca["ln_pob_2010"] = np.log(df_mca["pob_2010"].replace(0, np.nan))

print(f"\n  {'Variable':<30} ", end="")
for i in range(1, best_k + 1):
    print(f"{'C'+str(i)+' mean':>12} {'(SD)':>10}", end="")
print(f"  {'F':>8} {'p':>10}")
print("  " + "-" * (32 + best_k * 22 + 20))

for var, label in profile_vars:
    if var not in df_mca.columns:
        continue
    groups = [df_mca.loc[df_mca["mca_cluster"] == c, var].dropna()
              for c in range(1, best_k + 1)]
    # ANOVA
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*valid_groups)
    else:
        f_stat, p_val = np.nan, np.nan

    print(f"  {label:<30} ", end="")
    for g in groups:
        if len(g) > 0:
            print(f"{g.mean():>12.2f} ({g.std():>7.2f})", end="")
        else:
            print(f"{'---':>12} {'---':>10}", end="")
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
    if not np.isnan(f_stat):
        print(f"  {f_stat:>8.2f} {p_val:>10.4f} {sig}")
    else:
        print(f"  {'---':>8} {'---':>10}")

# ---
# Cross-tabulation: clusters x region
# ---

print("\nCross-tabulation: clusters x region...")

if "region" in df_mca.columns:
    ct = pd.crosstab(df_mca["mca_cluster"], df_mca["region"])
    print("\n  Counts:")
    print(ct.to_string())

    ct_pct = pd.crosstab(df_mca["mca_cluster"], df_mca["region"], normalize="index") * 100
    print("\n  Row percentages:")
    print(ct_pct.round(1).to_string())

    # Chi-squared test
    chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
    print(f"\n  Chi-squared = {chi2:.2f}, df = {dof}, p = {p_chi:.4f}")
    cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"  Cramér's V = {cramers_v:.4f}")

# ---
# Cross-tabulation: clusters x ECI
# ---

print("\nCross-tabulation: clusters x ECI...")

eci_valid = df_mca[df_mca["eci_software"].notna()].copy()
print(f"  Departments with ECI: {len(eci_valid)}")

if len(eci_valid) > 0:
    # Distribution of ECI by cluster
    print(f"\n  {'Cluster':<10} {'N':>5} {'Mean ECI':>10} {'SD':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*58}")
    for c in sorted(eci_valid["mca_cluster"].unique()):
        grp = eci_valid.loc[eci_valid["mca_cluster"] == c, "eci_software"]
        print(f"  {c:<10} {len(grp):>5} {grp.mean():>10.3f} {grp.std():>8.3f} "
              f"{grp.median():>8.3f} {grp.min():>8.3f} {grp.max():>8.3f}")

    # ANOVA
    eci_groups = [grp["eci_software"].dropna().values
                  for _, grp in eci_valid.groupby("mca_cluster")]
    if len(eci_groups) >= 2:
        f_eci, p_eci = stats.f_oneway(*eci_groups)
        # Eta-squared
        ss_between = sum(len(g) * (g.mean() - eci_valid["eci_software"].mean()) ** 2
                         for g in eci_groups)
        ss_total = ((eci_valid["eci_software"] - eci_valid["eci_software"].mean()) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        print(f"\n  ANOVA: F = {f_eci:.3f}, p = {p_eci:.4f}")
        print(f"  Eta-squared = {eta_sq:.4f}")

    # Kruskal-Wallis
    h_stat, p_kw = stats.kruskal(*eci_groups)
    print(f"  Kruskal-Wallis: H = {h_stat:.3f}, p = {p_kw:.4f}")

    # Proportion of ECI-having departments by cluster
    ct_eci = pd.crosstab(df_mca["mca_cluster"], df_mca["eci_software"].notna())
    if True in ct_eci.columns:
        ct_eci["pct_with_eci"] = ct_eci[True] / ct_eci.sum(axis=1) * 100
        print(f"\n  Proportion with ECI by cluster:")
        for c in ct_eci.index:
            pct = ct_eci.loc[c, "pct_with_eci"] if "pct_with_eci" in ct_eci.columns else 0
            print(f"    Cluster {c}: {pct:.1f}%")

# ---
# Cluster labelling
# ---

print("\nCluster labelling...")

# Rank-based labelling: compute standardised profiles and assign unique labels
cluster_profiles = {}
for c in range(1, best_k + 1):
    mask = df_mca["mca_cluster"] == c
    cluster_profiles[c] = {
        "n": mask.sum(),
        "nbi": df_mca.loc[mask, "pct_nbi_2010"].mean(),
        "rad": df_mca.loc[mask, "rad_2014"].mean(),
        "sec": df_mca.loc[mask, "pct_jefe_sec_2010"].mean(),
        "pop": df_mca.loc[mask, "pob_2010"].mean(),
        "eci": df_mca.loc[mask, "eci_software"].mean(),
        "act": df_mca.loc[mask, "tasa_empleo_2010"].mean() if "tasa_empleo_2010" in df_mca.columns else 0,
        "uni": df_mca.loc[mask, "pct_jefe_uni_2010"].mean(),
    }

# Sort clusters by Dim1 mean (deprivation -> metropolitan axis)
dim1_order = sorted(range(1, best_k + 1),
                    key=lambda c: df_mca.loc[df_mca["mca_cluster"] == c, "mca_dim1"].mean())

# Assign labels based on rank position along the principal axis
label_pool = [
    "Metropolitan-University",
    "Metropolitan-Diversified",
    "Intermediate-Educated",
    "Semi-Urban-Active",
    "Semi-Rural-Transitional",
    "Peripheral-Deprived",
    "Deep-Peripheral",
]

cluster_labels = {}
used = set()
for rank, c in enumerate(dim1_order):
    p = cluster_profiles[c]
    # Use distinct logic combining multiple dimensions
    if p["nbi"] > 18 and p["sec"] < 20:
        label = "Peripheral-Deprived"
    elif p["uni"] > 5 and p["rad"] > 30 and p["nbi"] < 6:
        label = "Metropolitan-Core"
    elif p["uni"] > 5 and p["rad"] > 30:
        label = "Metropolitan-Diversified"
    elif p["rad"] > 20 and p["sec"] > 25:
        label = "Intermediate-Urban"
    elif p["sec"] > 24 and p["nbi"] < 6:
        label = "Pampeana-Educated"
    elif p["act"] > 68 and p["sec"] < 25:
        label = "Semi-Rural-Active"
    elif p["nbi"] > 10:
        label = "Semi-Peripheral"
    else:
        label = f"Type-{c}"

    # Ensure uniqueness
    base_label = label
    suffix = 2
    while label in used:
        label = f"{base_label}-{suffix}"
        suffix += 1
    used.add(label)

    cluster_labels[c] = label
    print(f"  Cluster {c} -> '{label}' (N={p['n']}, NBI={p['nbi']:.1f}%, "
          f"Rad={p['rad']:.1f}, Sec={p['sec']:.1f}%, Uni={p['uni']:.1f}%, ECI={p['eci']:.2f})")

df_mca["mca_cluster_label"] = df_mca["mca_cluster"].map(cluster_labels)

# ---
# Plot clusters in MCA space
# ---

print("\nPlotting clusters in MCA space...")

fig, ax = plt.subplots(1, 1, figsize=(12, 9))
colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]

for c in sorted(df_mca["mca_cluster"].unique()):
    mask = df_mca["mca_cluster"] == c
    label = cluster_labels.get(c, f"Cluster {c}")
    n = mask.sum()
    ax.scatter(
        df_mca.loc[mask, "mca_dim1"],
        df_mca.loc[mask, "mca_dim2"],
        c=colours[c - 1], s=20, alpha=0.6, label=f"{label} (n={n})"
    )

ax.set_xlabel("MCA Dimension 1", fontsize=11)
ax.set_ylabel("MCA Dimension 2", fontsize=11)
ax.set_title(f"Departmental Types in MCA Space (k = {best_k}, N = {len(df_mca)})", fontsize=13)
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
ax.legend(loc="best", fontsize=9)

plt.tight_layout()
fig.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_03_cah_mca_clusters.png"),
            dpi=300, bbox_inches="tight")
print("  Cluster plot saved: fig_cah_mca_clusters.png")
plt.close()

# ---
# Save to database
# ---

print("\nSaving to database...")

with engine.connect() as conn:
    # Drop existing columns
    for col in ["mca_cluster", "mca_cluster_label"]:
        conn.execute(text(f"ALTER TABLE art1.departamentos DROP COLUMN IF EXISTS {col}"))
    conn.commit()

    # Add columns
    conn.execute(text("ALTER TABLE art1.departamentos ADD COLUMN mca_cluster INTEGER"))
    conn.execute(text("ALTER TABLE art1.departamentos ADD COLUMN mca_cluster_label TEXT"))
    conn.commit()

    # Update values
    for _, row in df_mca[["dpto5", "mca_cluster", "mca_cluster_label"]].iterrows():
        conn.execute(text(
            f"UPDATE art1.departamentos SET mca_cluster = {row['mca_cluster']}, "
            f"mca_cluster_label = '{row['mca_cluster_label']}' "
            f"WHERE dpto5 = '{row['dpto5']}'"
        ))
    conn.commit()

print(f"  Cluster assignments saved for {len(df_mca)} departments")

# Verify
verify = pd.read_sql(
    "SELECT mca_cluster, mca_cluster_label, COUNT(*) as n "
    "FROM art1.departamentos WHERE mca_cluster IS NOT NULL "
    "GROUP BY mca_cluster, mca_cluster_label ORDER BY mca_cluster",
    engine
)
print(f"\n  Verification:")
for _, row in verify.iterrows():
    print(f"    Cluster {row['mca_cluster']} ({row['mca_cluster_label']}): N = {row['n']}")

print(f"\nDone. {len(df_mca)} departments, k={best_k}, silhouette={silhouette_scores[best_k]:.4f}")
