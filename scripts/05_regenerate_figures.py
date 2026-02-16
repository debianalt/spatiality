"""
Regenerate all article figures with uniform formatting (300 DPI).
Figures 1-6 (main text) and Figures S1-S2 (supplementary).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import prince
import sqlalchemy
from pathlib import Path
from sqlalchemy import text
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from adjustText import adjust_text
import matplotlib.transforms as transforms

# ---
# Style
# ---

STYLE = {
    "title_size": 14,
    "axis_label_size": 12,
    "tick_size": 11,
    "annot_size": 10,
    "legend_size": 10,
    "small_annot_size": 9,
    "dpi": 300,
    "fig_main": (12, 8),
    "fig_wide": (14, 6),
    "fig_tall": (10, 10),
}

# Cluster colours — consistent across all figures
CLUSTER_COLOURS = {
    1: "#e41a1c",  # red
    2: "#377eb8",  # blue
    3: "#4daf4a",  # green
    4: "#984ea3",  # purple
    5: "#ff7f00",  # orange
    6: "#a65628",  # brown
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": STYLE["tick_size"],
    "axes.titlesize": STYLE["title_size"],
    "axes.labelsize": STYLE["axis_label_size"],
    "xtick.labelsize": STYLE["tick_size"],
    "ytick.labelsize": STYLE["tick_size"],
    "legend.fontsize": STYLE["legend_size"],
    "figure.dpi": 100,
    "savefig.dpi": STYLE["dpi"],
    "savefig.bbox": "tight",
})

OUT_DIR = str(Path(__file__).resolve().parent.parent / "figures")

# ---

engine = sqlalchemy.create_engine("postgresql://postgres:postgres@localhost/posadas")
df = pd.read_sql("SELECT * FROM art1.departamentos", engine)
df["ln_pob_2010"] = np.log(df["pob_2010"].replace(0, np.nan))
print(f"  Loaded {len(df)} departments")

# Cluster labels
cluster_labels = {}
if "mca_cluster_label" in df.columns:
    for c in df["mca_cluster"].dropna().unique():
        label = df.loc[df["mca_cluster"] == c, "mca_cluster_label"].iloc[0]
        cluster_labels[int(c)] = label
    print(f"  Cluster labels: {cluster_labels}")

# ---
# MCA refit for biplot coordinates
# ---

active_vars = [
    "pct_jefe_sec_2010", "pct_jefe_uni_2010", "pct_pc_2010", "rad_2014",
    "tasa_empleo_2010","pct_nbi_2010", "pct_hacinam_2010", "ln_pob_2010",
]

var_labels_mca = {
    "pct_jefe_sec_2010": "Sec. educ.",
    "pct_jefe_uni_2010": "Univ. educ.",
    "pct_pc_2010": "PC ownership",
    "rad_2014": "Radiance",
    "tasa_empleo_2010": "Empl. rate",
    "pct_nbi_2010": "Poverty (NBI)",
    "pct_hacinam_2010": "Overcrowding",
    "ln_pob_2010": "ln(Pop.)",
}

tercile_labels = ["Low", "Mid", "High"]
df_active = df.dropna(subset=active_vars).copy()

for var in active_vars:
    col_name = f"{var}_t"
    try:
        df_active[col_name] = pd.qcut(df_active[var], q=3, labels=tercile_labels, duplicates="drop")
    except ValueError:
        df_active[col_name] = pd.cut(df_active[var], bins=3, labels=tercile_labels)
    short = var_labels_mca.get(var, var)
    df_active[col_name] = df_active[col_name].astype(str).apply(lambda x, s=short: f"{s}_{x}")

active_cols_t = [f"{v}_t" for v in active_vars]
X_mca = df_active[active_cols_t].copy()

mca = prince.MCA(n_components=10, random_state=42)
mca = mca.fit(X_mca)

eigenvalues = mca.eigenvalues_
total_inertia = mca.total_inertia_
pct_inertia = [e / total_inertia * 100 for e in eigenvalues]

# Benzecri correction
K = len(active_vars)
threshold = 1.0 / K
benz_eigenvalues = []
for ev in eigenvalues:
    if ev > threshold:
        corrected = ((K / (K - 1)) * (ev - threshold)) ** 2
        benz_eigenvalues.append(corrected)
    else:
        benz_eigenvalues.append(0.0)
benz_total = sum(benz_eigenvalues)
benz_pct = [b / benz_total * 100 if benz_total > 0 else 0 for b in benz_eigenvalues]
n_retained = sum(1 for b in benz_eigenvalues if b > 0)

row_coords = mca.row_coordinates(X_mca)
row_coords.columns = [f"mca_dim{i+1}" for i in range(row_coords.shape[1])]
row_coords.index = df_active.index
for col in row_coords.columns:
    df_active[col] = row_coords[col]

col_coords = mca.column_coordinates(X_mca)
col_coords.columns = [f"Dim{i+1}" for i in range(col_coords.shape[1])]

contribs = mca.column_contributions_
if hasattr(contribs, 'columns'):
    contribs.columns = [f"Dim{i+1}" for i in range(contribs.shape[1])]

print(f"  MCA refit: {len(df_active)} departments, {n_retained} axes retained")


def draw_confidence_ellipse(x, y, ax, n_std=1.5, facecolor='none', **kwargs):
    """Concentration ellipse (n_std=1.5 ~ 78% of points)."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                       width=ell_radius_x * 2,
                       height=ell_radius_y * 2,
                       facecolor=facecolor,
                       **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# ---
# Figure 1: ECI vs developer density
# ---

fig1, ax1 = plt.subplots(figsize=STYLE["fig_main"])

eci_df = df[df["eci_software"].notna() & df["mca_cluster"].notna()].copy()

# Province abbreviations for disambiguation
PROV_ABBREV = {
    "Buenos Aires": "BA", "Catamarca": "Cat.", "Chaco": "Cha.", "Chubut": "Chu.",
    "Córdoba": "Cba.", "Corrientes": "Ctes.", "Entre Ríos": "ER",
    "Formosa": "Fsa.", "Jujuy": "Juj.", "La Pampa": "LP", "La Rioja": "LR",
    "Mendoza": "Mza.", "Misiones": "Mis.", "Neuquén": "Nqn.",
    "Río Negro": "RN", "Salta": "Sal.", "San Juan": "SJ", "San Luis": "SL",
    "Santa Cruz": "SC", "Santa Fe": "SF", "Santiago del Estero": "SdE",
    "Tierra del Fuego": "TdF", "Tucumán": "Tuc.",
    "Ciudad Autónoma de Buenos Aires": "CABA",
}

def dept_label(row):
    """Build display label: add province abbreviation for ambiguous names."""
    name = row.get("departamento", "")
    prov = row.get("provincia", "")
    if name.lower() in ("capital", "la capital"):
        abbr = PROV_ABBREV.get(prov, prov[:3])
        return f"{name} ({abbr})"
    return name

# --- Scatter by cluster type ---
xlim_max = 72
for c in sorted(eci_df["mca_cluster"].unique()):
    c_int = int(c)
    mask = eci_df["mca_cluster"] == c
    label = cluster_labels.get(c_int, f"Type {c_int}")
    n = mask.sum()
    ax1.scatter(
        eci_df.loc[mask, "gh_devs_per_10k"],
        eci_df.loc[mask, "eci_software"],
        c=CLUSTER_COLOURS.get(c_int, "grey"), s=40, alpha=0.7,
        label=f"{label} (n={n})", edgecolors="white", linewidths=0.3, zorder=2
    )

# --- Top labels: high-ECI or high-devs departments ---
notable_top = eci_df[
    (eci_df["eci_software"] > 1.6) | (eci_df["gh_devs_per_10k"] > 50)
].copy()

texts_top = []
for _, row in notable_top.iterrows():
    name = dept_label(row)
    if name:
        texts_top.append(ax1.text(
            row["gh_devs_per_10k"], row["eci_software"], name,
            fontsize=STYLE["small_annot_size"], fontstyle="italic",
        ))
print(f"  Top labels: {len(texts_top)}")

if texts_top:
    adjust_text(
        texts_top, ax=ax1,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        force_text=(4.0, 4.0),
        force_points=(2.5, 2.5),
        expand_text=(2.2, 2.2),
        expand_points=(2.0, 2.0),
        only_move={"points": "y", "text": "xy"},
        iterations=1500,
    )

# --- Bottom labels: stack on the right side ---
notable_bottom = eci_df[
    (eci_df["gh_devs_per_10k"] <= xlim_max) &
    (eci_df["eci_software"] < -1.8)
].copy()

if len(notable_bottom) > 0:
    notable_bottom = notable_bottom.sort_values("eci_software", ascending=True)
    n_bot = len(notable_bottom)
    y_min = notable_bottom["eci_software"].min() + 0.1
    y_max = notable_bottom["eci_software"].max() + 0.7
    y_positions = np.linspace(y_min, y_max, n_bot)
    for i, (_, row) in enumerate(notable_bottom.iterrows()):
        name = dept_label(row)
        if name:
            ax1.annotate(
                name,
                xy=(row["gh_devs_per_10k"], row["eci_software"]),
                xytext=(22, y_positions[i]),
                fontsize=STYLE["small_annot_size"], fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
                ha="left", va="center",
            )

ax1.set_xlim(-2, xlim_max)
ax1.set_ylim(-3.0, 2.8)
ax1.set_xlabel("Developers per 10,000 inhabitants")
ax1.set_ylabel("ECIsoftware")
ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax1.legend(loc="lower right", framealpha=0.9)

fig1.savefig(f"{OUT_DIR}/fig_eci_vs_devs.png", dpi=STYLE["dpi"], bbox_inches="tight")
print("  Saved: fig_eci_vs_devs.png")
plt.close(fig1)


# ---
# Figure 2: PCI vs ubiquity
# ---

# Reconstruct PCI from bipartite network — matching analysis_41_validation.py exactly
from numpy.linalg import eig

pci_q = ("SELECT LEFT(redcode, 5) AS dpto5, primary_language, COUNT(*) AS repos "
         "FROM github_argentina.repos "
         "WHERE primary_language IS NOT NULL AND primary_language != '' "
         "GROUP BY 1, 2")
df_raw = pd.read_sql(pci_q, engine)
print(f"  Raw records: {len(df_raw)}")

# Department mapping (matching original analysis)
EXCLUDE = {"94021", "94028"}
MAPD = {"06217": "06218", "06466": "06218", "94007": "94008", "94014": "94015", "94011": "94015"}
df_raw["dpto5"] = df_raw["dpto5"].apply(lambda x: "02000" if x.startswith("02") else MAPD.get(x, x))
df_raw = df_raw[~df_raw["dpto5"].isin(EXCLUDE)].copy()
df_agg = df_raw.groupby(["dpto5", "primary_language"], as_index=False)["repos"].sum()

# Filter: departments >= 10 repos, languages >= 30 repos
dt = df_agg.groupby("dpto5")["repos"].sum()
vd = dt[dt >= 10].index
lt = df_agg.groupby("primary_language")["repos"].sum()
vl = lt[lt >= 30].index
df_f = df_agg[df_agg["dpto5"].isin(vd) & df_agg["primary_language"].isin(vl)]

Mat = df_f.pivot_table(index="dpto5", columns="primary_language", values="repos", fill_value=0)
print(f"  Bipartite matrix: {Mat.shape[0]} departments × {Mat.shape[1]} languages")

# RCA
Mv = Mat.values.astype(float)
rs = Mv.sum(axis=1, keepdims=True)
cs = Mv.sum(axis=0, keepdims=True)
tot = Mv.sum()
RCA = (Mv / rs) / (cs / tot)
Amat = (RCA >= 1).astype(float)

div_vec = Amat.sum(axis=1)
ubiq_vec = Amat.sum(axis=0)
div_vec[div_vec == 0] = 1e-10
ubiq_vec[ubiq_vec == 0] = 1e-10

# Eigenvalue decomposition (second eigenvector of normalised adjacency)
Di = np.diag(1.0 / div_vec)
Ui = np.diag(1.0 / ubiq_vec)
Mt_lang = Ui @ Amat.T @ Di @ Amat
ev, evec = eig(Mt_lang)
idx_s = np.argsort(-ev.real)
evec = evec[:, idx_s]

pci_raw = evec[:, 1].real
pci_vals = (pci_raw - pci_raw.mean()) / pci_raw.std()

# Fix sign: PCI should correlate negatively with ubiquity
if np.corrcoef(pci_vals, ubiq_vec)[0, 1] > 0:
    pci_vals = -pci_vals
    print(f"  PCI sign flipped (now negatively correlated with ubiquity)")

lang_counts_pci = lt[Mat.columns]
lang_stats = pd.DataFrame({
    "language": Mat.columns,
    "pci": pci_vals,
    "ubiquity": ubiq_vec,
    "total_repos": lang_counts_pci.values,
})

print(f"  Languages with PCI: {len(lang_stats)}")
print(f"  Top 5 PCI: {lang_stats.nlargest(5, 'pci')[['language', 'pci', 'ubiquity']].to_string(index=False)}")
print(f"  Bottom 5 PCI: {lang_stats.nsmallest(5, 'pci')[['language', 'pci', 'ubiquity']].to_string(index=False)}")

# Plot
fig2, ax2 = plt.subplots(figsize=STYLE["fig_main"])

# Bubble size proportional to total repos (log scale)
sizes = np.log1p(lang_stats["total_repos"]) * 15

# Colour by PCI
norm = mcolors.TwoSlopeNorm(vmin=lang_stats["pci"].min(), vcenter=0,
                             vmax=lang_stats["pci"].max())
cmap = plt.cm.RdYlGn

sc2 = ax2.scatter(
    lang_stats["ubiquity"], lang_stats["pci"],
    s=sizes, c=lang_stats["pci"], cmap=cmap, norm=norm,
    alpha=0.7, edgecolors="grey", linewidths=0.5, zorder=2
)

cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.02)
cbar2.set_label("Product Complexity Index (PCI)", fontsize=STYLE["axis_label_size"])
cbar2.ax.tick_params(labelsize=STYLE["tick_size"])

# Label languages — select top/bottom by PCI plus notable ones
top_pci = lang_stats.nlargest(8, "pci")["language"].tolist()
bottom_pci = lang_stats.nsmallest(5, "pci")["language"].tolist()
notable_langs = ["Python", "R", "C", "C++", "Go", "Rust", "Haskell"]
label_langs = list(set(top_pci + bottom_pci + [l for l in notable_langs if l in lang_stats["language"].values]))

texts2 = []
for _, row in lang_stats.iterrows():
    if row["language"] in label_langs:
        texts2.append(ax2.text(
            row["ubiquity"], row["pci"], row["language"],
            fontsize=STYLE["annot_size"], fontweight="bold",
            ha="center", va="center"
        ))

# Use adjustText to prevent overlap
if texts2:
    adjust_text(
        texts2, ax=ax2,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        force_text=(1.5, 1.5),
        force_points=(1.0, 1.0),
        expand_text=(1.4, 1.6),
        expand_points=(1.3, 1.3),
        only_move={"points": "y", "text": "xy"},
    )

ax2.set_xlabel("Ubiquity (number of departments with RCA $\\geq$ 1)")
ax2.set_ylabel("Product Complexity Index (PCI)")
ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")

fig2.savefig(f"{OUT_DIR}/fig_pci_ubiquity.png", dpi=STYLE["dpi"], bbox_inches="tight")
print("  Saved: fig_pci_ubiquity.png")
plt.close(fig2)


# ---
# Figure 3: MCA biplot
# ---

fig3, ax3 = plt.subplots(figsize=(13, 9))

# Row points coloured by ECI gradient
has_eci = df_active["eci_software"].notna()
no_eci = ~has_eci

ax3.scatter(
    df_active.loc[no_eci, "mca_dim1"],
    df_active.loc[no_eci, "mca_dim2"],
    c="lightgrey", s=15, alpha=0.4, zorder=1, label="No ECI"
)

if has_eci.any():
    eci_vals_plot = df_active.loc[has_eci, "eci_software"]
    sc3 = ax3.scatter(
        df_active.loc[has_eci, "mca_dim1"],
        df_active.loc[has_eci, "mca_dim2"],
        c=eci_vals_plot, cmap="RdYlGn", s=25, alpha=0.7, zorder=2,
        vmin=-2.5, vmax=2.5, edgecolors="none"
    )
    cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.6, pad=0.02)
    cbar3.set_label("ECIsoftware (supplementary)", fontsize=STYLE["axis_label_size"])
    cbar3.ax.tick_params(labelsize=STYLE["tick_size"])

# Modality labels + region centroids — all handled by adjustText together
# so that no label overlaps any triangle marker or square centroid marker
all_anchor_x = []
all_anchor_y = []
texts3 = []

# Active modality labels (triangle markers)
for idx in col_coords.index:
    d1 = col_coords.loc[idx, "Dim1"]
    d2 = col_coords.loc[idx, "Dim2"]
    raw = str(idx)
    label = raw.split("__", 1)[1] if "__" in raw else raw
    # Replace last underscore (before Low/Mid/High) with ": " for readability
    for level in ["_Low", "_Mid", "_High"]:
        if label.endswith(level):
            label = label[:-len(level)] + ": " + level[1:]
            break
    ax3.plot(d1, d2, "k^", markersize=8, zorder=3)
    all_anchor_x.append(d1)
    all_anchor_y.append(d2)
    texts3.append(ax3.text(
        d1, d2, label,
        fontsize=STYLE["annot_size"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.85),
        zorder=4
    ))

# Region centroids (supplementary) — included in adjustText to prevent overlaps
if "region" in df_active.columns:
    region_centroids = df_active.groupby("region")[["mca_dim1", "mca_dim2"]].mean()
    for reg in region_centroids.index:
        rx = region_centroids.loc[reg, "mca_dim1"]
        ry = region_centroids.loc[reg, "mca_dim2"]
        ax3.plot(rx, ry, "s", markersize=12, color="navy", zorder=5)
        all_anchor_x.append(rx)
        all_anchor_y.append(ry)
        texts3.append(ax3.text(
            rx, ry, reg,
            fontsize=STYLE["annot_size"], fontweight="bold", color="navy",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="navy", alpha=0.85),
            zorder=6
        ))

if texts3:
    adjust_text(
        texts3, ax=ax3,
        x=all_anchor_x, y=all_anchor_y,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        force_text=(1.5, 1.5),
        force_points=(2.0, 2.0),
        expand_text=(1.5, 1.5),
        expand_points=(2.0, 2.0),
        iterations=400,
    )

pct1 = pct_inertia[0]
pct2 = pct_inertia[1]
ax3.set_xlabel(f"Dimension 1 ({pct1:.1f}% inertia)")
ax3.set_ylabel(f"Dimension 2 ({pct2:.1f}% inertia)")
ax3.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax3.axvline(0, color="grey", linewidth=0.5, linestyle="--")

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
           markersize=8, label="No ECI (supplementary)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
           markersize=8, label="Active modalities"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="navy",
           markersize=8, label="Region centroids (suppl.)"),
]
ax3.legend(handles=legend_elements, loc="lower right")

fig3.savefig(f"{OUT_DIR}/fig_mca_biplot.png", dpi=STYLE["dpi"], bbox_inches="tight")
print("  Saved: fig_mca_biplot.png")
plt.close(fig3)


# ---
# Figure 4: dendrogram
# ---

# Re-compute linkage
mca_cols = [c for c in df.columns if c.startswith("mca_dim")]
df_mca = df.dropna(subset=mca_cols).copy()
X_cah = df_mca[mca_cols].values
dist_condensed = pdist(X_cah, metric="euclidean")
Z_linkage = linkage(dist_condensed, method="ward")

best_k = df_mca["mca_cluster"].nunique()
print(f"  k = {best_k}")

fig4, ax4 = plt.subplots(figsize=STYLE["fig_wide"])
dendrogram(
    Z_linkage,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=STYLE["tick_size"],
    color_threshold=Z_linkage[-(best_k - 1), 2],
    ax=ax4,
    above_threshold_color="grey"
)
ax4.set_xlabel("Cluster size")
ax4.set_ylabel("Ward distance")
ax4.axhline(Z_linkage[-(best_k - 1), 2], color="red", linestyle="--", linewidth=1.5,
            label=f"Cut at k = {best_k}")
ax4.legend()

fig4.savefig(f"{OUT_DIR}/fig_cah_dendrogram.png", dpi=STYLE["dpi"], bbox_inches="tight")
print("  Saved: fig_cah_dendrogram.png")
plt.close(fig4)


# ---
# Figure 5: clusters in MCA space
# ---

fig5, ax5 = plt.subplots(figsize=STYLE["fig_main"])

for c in sorted(df_mca["mca_cluster"].unique()):
    c_int = int(c)
    mask = df_mca["mca_cluster"] == c
    label = cluster_labels.get(c_int, f"Cluster {c_int}")
    n = mask.sum()
    colour = CLUSTER_COLOURS.get(c_int, "grey")

    x_vals = df_mca.loc[mask, "mca_dim1"].values
    y_vals = df_mca.loc[mask, "mca_dim2"].values

    # Scatter points
    ax5.scatter(
        x_vals, y_vals,
        c=colour, s=25, alpha=0.5, label=f"{label} (n={n})",
        edgecolors="white", linewidths=0.3, zorder=2
    )

    # Concentration ellipse (1.5 SD ~ 78% of points)
    draw_confidence_ellipse(
        x_vals, y_vals, ax5, n_std=1.5,
        edgecolor=colour, linewidth=2.0, linestyle="-", alpha=0.9, zorder=3
    )

    # Centroid marker
    cx, cy = np.mean(x_vals), np.mean(y_vals)
    ax5.plot(cx, cy, marker="+", color=colour, markersize=14,
             markeredgewidth=2.5, zorder=4)

ax5.set_xlabel(f"Dimension 1 ({pct_inertia[0]:.1f}% inertia)")
ax5.set_ylabel(f"Dimension 2 ({pct_inertia[1]:.1f}% inertia)")
ax5.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax5.axvline(0, color="grey", linewidth=0.5, linestyle="--")
ax5.legend(loc="best")

fig5.savefig(f"{OUT_DIR}/fig_cah_mca_clusters.png", dpi=STYLE["dpi"], bbox_inches="tight")
print("  Saved: fig_cah_mca_clusters.png")
plt.close(fig5)


# ---
# Figure 6: forest plot
# ---

predictors = ["ln_pob_2010", "pct_jefe_sec_2010", "pct_pc_2010", "pct_nbi_2010", "rad_2014"]
dv = "eci_software"

var_labels_reg = {
    "ln_pob_2010": "ln(Population)",
    "pct_jefe_sec_2010": "Secondary education (%)",
    "pct_pc_2010": "Computer ownership (%)",
    "pct_nbi_2010": "Structural poverty (%)",
    "rad_2014": "Nighttime radiance",
}

def compute_std_betas(data, preds, dv_name):
    cols = preds + [dv_name]
    clean = data.dropna(subset=cols)
    if len(clean) < len(preds) + 2:
        return None
    X = clean[preds].copy()
    y = clean[dv_name].copy()
    y_std = (y - y.mean()) / y.std()
    for c in preds:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = add_constant(X)
    model = sm.OLS(y_std, X).fit(cov_type="HC1")
    result = {}
    for var in preds:
        result[var] = {
            "beta": model.params[var],
            "se": model.bse[var],
            "p": model.pvalues[var],
            "ci_low": model.conf_int().loc[var, 0],
            "ci_high": model.conf_int().loc[var, 1],
        }
    return result

# Pooled
pooled_data = df[df["eci_software"].notna()].copy()
pooled_betas = compute_std_betas(pooled_data, predictors, dv)

# Per type
type_betas = {}
clusters_sorted = sorted(df["mca_cluster"].dropna().unique())
for c in clusters_sorted:
    c_int = int(c)
    subset = df[df["mca_cluster"] == c].copy()
    n_eci = subset[dv].notna().sum()
    if n_eci >= 30:
        type_betas[c_int] = compute_std_betas(subset, predictors, dv)

# Build plot data
plot_data = []
if pooled_betas:
    for var in predictors:
        if var in pooled_betas:
            plot_data.append({
                "variable": var_labels_reg.get(var, var),
                "type": "Pooled",
                "beta": pooled_betas[var]["beta"],
                "ci_low": pooled_betas[var]["ci_low"],
                "ci_high": pooled_betas[var]["ci_high"],
                "p": pooled_betas[var]["p"],
            })

for c in clusters_sorted:
    c_int = int(c)
    label_c = cluster_labels.get(c_int, f"Type {c_int}")
    betas = type_betas.get(c_int)
    if betas:
        for var in predictors:
            if var in betas:
                plot_data.append({
                    "variable": var_labels_reg.get(var, var),
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

    fig6, ax6 = plt.subplots(figsize=(12, max(7, n_vars * 1.8)))

    forest_colours = ["black"] + [CLUSTER_COLOURS.get(int(c), "grey") for c in clusters_sorted
                                   if int(c) in type_betas]
    offsets = np.linspace(-0.3, 0.3, n_types)

    for j, type_name in enumerate(types):
        sub = plot_df[plot_df["type"] == type_name]
        colour = forest_colours[j % len(forest_colours)]
        for _, row in sub.iterrows():
            var_idx = variables.index(row["variable"])
            y_pos = n_vars - 1 - var_idx + offsets[j]
            marker = "D" if type_name == "Pooled" else "o"
            size = 12 if type_name == "Pooled" else 10

            ax6.errorbar(
                row["beta"], y_pos,
                xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
                fmt=marker, color=colour, markersize=size, capsize=5, capthick=2.0,
                linewidth=2.5, alpha=0.85
            )

    ax6.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax6.set_yticks(range(n_vars))
    ax6.set_yticklabels(list(reversed(variables)))
    ax6.set_xlabel("Standardised coefficient ($\\beta$) with 95% CI")

    # Legend
    legend_handles = []
    for j, type_name in enumerate(types):
        colour = forest_colours[j % len(forest_colours)]
        marker = "D" if type_name == "Pooled" else "o"
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor=colour,
                       markersize=11, label=type_name)
        )
    ax6.legend(handles=legend_handles, loc="best")

    fig6.savefig(f"{OUT_DIR}/fig_forest_plot.png", dpi=STYLE["dpi"], bbox_inches="tight")
    print("  Saved: fig_forest_plot.png")
    plt.close(fig6)


# ---
# Figure S2: MCA + CAH diagnostics
# ---

labels_final = df_mca["mca_cluster"].values.astype(int)

# Compute silhouette scores for k range
k_range = range(3, 8)
sil_scores = {}
ch_scores = {}
for k in k_range:
    labs_k = fcluster(Z_linkage, t=k, criterion="maxclust")
    sil_scores[k] = silhouette_score(X_cah, labs_k)
    from sklearn.metrics import calinski_harabasz_score
    ch_scores[k] = calinski_harabasz_score(X_cah, labs_k)

fig_s1, axes = plt.subplots(2, 2, figsize=(14, 11))
fig_s1.subplots_adjust(hspace=0.35, wspace=0.35)

# --- (a) Raw eigenvalue scree plot ---
ax_a = axes[0, 0]
ax_a.bar(range(1, len(eigenvalues)+1), eigenvalues, color="steelblue", alpha=0.8)
ax_a.axhline(threshold, color="red", linestyle="--", label=f"1/K = {threshold:.3f}")
ax_a.set_xlabel("Axis")
ax_a.set_ylabel("Eigenvalue")
ax_a.set_title("(a) Eigenvalue scree plot")
ax_a.legend(fontsize=STYLE["small_annot_size"])

# --- (b) Benzecri-corrected eigenvalues ---
ax_b = axes[0, 1]
benz_nz = [b for b in benz_eigenvalues if b > 0]
benz_pct_nz = [b / benz_total * 100 for b in benz_nz]
ax_b.bar(range(1, len(benz_nz)+1), benz_nz, color="darkorange", alpha=0.8)
ax_b_twin = ax_b.twinx()
ax_b_twin.plot(range(1, len(benz_nz)+1), np.cumsum(benz_pct_nz), "ro-", markersize=7)
ax_b_twin.set_ylabel("Cumulative %", fontsize=STYLE["annot_size"])
ax_b_twin.tick_params(labelsize=STYLE["small_annot_size"])
ax_b.set_xlabel("Axis")
ax_b.set_ylabel("Corrected eigenvalue")
ax_b.set_title("(b) Benzecri-corrected eigenvalues")

# --- (c) Cluster quality metrics ---
ax_c = axes[1, 0]
ks = list(k_range)
sils = [sil_scores[k] for k in ks]
chs = [ch_scores[k] for k in ks]
ax_c.plot(ks, sils, "bo-", markersize=8, linewidth=2, label="Silhouette")
ax_c.set_xlabel("Number of clusters (k)")
ax_c.set_ylabel("Silhouette score", color="blue")
ax_c.tick_params(axis="y", labelcolor="blue")
ax_c_twin = ax_c.twinx()
ax_c_twin.plot(ks, chs, "rs--", markersize=8, linewidth=2, label="Calinski-Harabasz")
ax_c_twin.set_ylabel("Calinski-Harabasz", color="red", fontsize=STYLE["annot_size"])
ax_c_twin.tick_params(axis="y", labelcolor="red", labelsize=STYLE["small_annot_size"])
ax_c.axvline(best_k, color="green", linestyle=":", linewidth=2, label=f"k = {best_k}")
ax_c.set_title("(c) Cluster quality metrics")
ax_c.legend(loc="upper left", fontsize=STYLE["small_annot_size"])

# --- (d) Silhouette per sample ---
ax_d = axes[1, 1]
sample_sil = silhouette_samples(X_cah, labels_final)
y_lower = 10
for i in range(1, best_k + 1):
    cluster_sil = np.sort(sample_sil[labels_final == i])
    size_i = len(cluster_sil)
    y_upper = y_lower + size_i
    colour = CLUSTER_COLOURS.get(i, cm.nipy_spectral(float(i) / (best_k + 1)))
    ax_d.fill_betweenx(
        np.arange(y_lower, y_upper), 0, cluster_sil,
        facecolor=colour, edgecolor=colour, alpha=0.7
    )
    ax_d.text(-0.05, y_lower + 0.5 * size_i, f"C{i}",
              fontsize=STYLE["small_annot_size"], fontweight="bold")
    y_lower = y_upper + 10

ax_d.axvline(sil_scores[best_k], color="red", linestyle="--", linewidth=1.5)
ax_d.set_xlabel("Silhouette coefficient")
ax_d.set_ylabel("Departments (sorted)")
ax_d.set_title(f"(d) Silhouette plot (k = {best_k})")

fig_s1.savefig(f"{OUT_DIR}/fig_diagnostics_panel.png", dpi=STYLE["dpi"], bbox_inches="tight")
print("  Saved: fig_diagnostics_panel.png")
plt.close(fig_s1)


