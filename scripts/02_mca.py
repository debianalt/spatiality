"""
Multiple Correspondence Analysis on 511 Argentine departments.
8 active variables discretised into terciles (24 modalities), Benzecri
correction, 5 retained axes. Supplementary projections of ECI and regions.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import prince
import sqlalchemy
from pathlib import Path
from sqlalchemy import text
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---

print("Running MCA...")

engine = sqlalchemy.create_engine("postgresql://postgres:postgres@localhost/posadas")
df = pd.read_sql("SELECT * FROM art1.departamentos", engine)
print(f"\nLoaded {len(df)} departments, {df.shape[1]} columns")

# ---
# Variable construction and discretisation
# ---

df["ln_pob_2010"] = np.log(df["pob_2010"].replace(0, np.nan))

# 8 active variables — continuous originals
active_vars = [
    "pct_jefe_sec_2010",
    "pct_jefe_uni_2010",
    "pct_pc_2010",
    "rad_2014",
    "tasa_empleo_2010",
    "pct_nbi_2010",
    "pct_hacinam_2010",
    "ln_pob_2010",
]

var_labels = {
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

# Drop rows with missing values in active variables
df_active = df.dropna(subset=active_vars).copy()
print(f"  Departments with complete active variables: {len(df_active)}")

# Discretise each variable into terciles
for var in active_vars:
    col_name = f"{var}_t"
    try:
        df_active[col_name] = pd.qcut(
            df_active[var], q=3, labels=tercile_labels, duplicates="drop"
        )
    except ValueError:
        # If terciles fail (e.g. too many ties), use fixed bins
        df_active[col_name] = pd.cut(
            df_active[var], bins=3, labels=tercile_labels
        )
    short = var_labels.get(var, var)
    df_active[col_name] = df_active[col_name].astype(str).apply(
        lambda x, s=short: f"{s}_{x}"
    )
    print(f"  {var} -> {col_name}: {df_active[col_name].value_counts().to_dict()}")

active_cols_t = [f"{v}_t" for v in active_vars]

# ---
# MCA fit
# ---

X = df_active[active_cols_t].copy()
print(f"  Input matrix: {X.shape[0]} rows × {X.shape[1]} columns")
print(f"  Total modalities: {sum(X[c].nunique() for c in X.columns)}")

mca = prince.MCA(n_components=10, random_state=42)
mca = mca.fit(X)

# Eigenvalues and inertia
eigenvalues = mca.eigenvalues_
total_inertia = mca.total_inertia_
pct_inertia = [e / total_inertia * 100 for e in eigenvalues]
cum_inertia = np.cumsum(pct_inertia)

print(f"\n  Total inertia: {total_inertia:.4f}")
print(f"\n  {'Axis':<6} {'Eigenvalue':<12} {'% Inertia':<12} {'Cumulative %':<14}")
print(f"  {'-'*44}")
for i, (ev, pct, cpct) in enumerate(zip(eigenvalues, pct_inertia, cum_inertia)):
    print(f"  {i+1:<6} {ev:<12.4f} {pct:<12.2f} {cpct:<14.2f}")

# Benzecri correction (for K categories per variable, p active variables)
K = len(active_vars)  # number of active variables (8)
J = sum(X[c].nunique() for c in X.columns)  # total modalities
threshold = 1.0 / K  # only correct eigenvalues > 1/K

print(f"\n  Benzecri correction (threshold = 1/{K} = {threshold:.4f}):")
benz_eigenvalues = []
for ev in eigenvalues:
    if ev > threshold:
        corrected = ((K / (K - 1)) * (ev - threshold)) ** 2
        benz_eigenvalues.append(corrected)
    else:
        benz_eigenvalues.append(0.0)

benz_total = sum(benz_eigenvalues)
benz_pct = [b / benz_total * 100 if benz_total > 0 else 0 for b in benz_eigenvalues]
benz_cum = np.cumsum(benz_pct)
n_retained = sum(1 for b in benz_eigenvalues if b > 0)

print(f"  {'Axis':<6} {'Corrected EV':<14} {'% Corrected':<14} {'Cumulative %':<14}")
print(f"  {'-'*48}")
for i, (bev, bpct, bcpct) in enumerate(zip(benz_eigenvalues, benz_pct, benz_cum)):
    marker = " *" if bev > 0 else ""
    print(f"  {i+1:<6} {bev:<14.6f} {bpct:<14.2f} {bcpct:<14.2f}{marker}")

print(f"\n  Axes retained (Benzecri EV > 0): {n_retained}")

# ---
# Row coordinates
# ---

row_coords = mca.row_coordinates(X)
row_coords.columns = [f"mca_dim{i+1}" for i in range(row_coords.shape[1])]
row_coords.index = df_active.index

print(f"  Row coordinates computed: {row_coords.shape}")
print(f"  Dim1 range: [{row_coords['mca_dim1'].min():.3f}, {row_coords['mca_dim1'].max():.3f}]")
print(f"  Dim2 range: [{row_coords['mca_dim2'].min():.3f}, {row_coords['mca_dim2'].max():.3f}]")

# Merge back to df_active
for col in row_coords.columns:
    df_active[col] = row_coords[col]

# ---
# Column coordinates and contributions
# ---

col_coords = mca.column_coordinates(X)
col_coords.columns = [f"Dim{i+1}" for i in range(col_coords.shape[1])]

# Contributions
contribs = mca.column_contributions_
if hasattr(contribs, 'columns'):
    contribs.columns = [f"Dim{i+1}" for i in range(contribs.shape[1])]

print(f"\n  Modality coordinates (Dim1, Dim2):")
print(f"  {'Modality':<30} {'Dim1':>8} {'Dim2':>8} {'Ctr1%':>8} {'Ctr2%':>8}")
print(f"  {'-'*62}")
for idx in col_coords.index:
    d1 = col_coords.loc[idx, "Dim1"]
    d2 = col_coords.loc[idx, "Dim2"]
    c1 = contribs.loc[idx, "Dim1"] * 100 if hasattr(contribs, 'loc') else 0
    c2 = contribs.loc[idx, "Dim2"] * 100 if hasattr(contribs, 'loc') else 0
    print(f"  {str(idx):<30} {d1:>8.3f} {d2:>8.3f} {c1:>8.2f} {c2:>8.2f}")

# Top contributions to Dim1
print(f"\n  Top 8 contributors to Dim1:")
if hasattr(contribs, 'loc'):
    top_dim1 = contribs["Dim1"].sort_values(ascending=False).head(8)
    for mod, ctr in top_dim1.items():
        print(f"    {str(mod):<30} {ctr*100:.2f}%")

print(f"\n  Top 8 contributors to Dim2:")
if hasattr(contribs, 'loc'):
    top_dim2 = contribs["Dim2"].sort_values(ascending=False).head(8)
    for mod, ctr in top_dim2.items():
        print(f"    {str(mod):<30} {ctr*100:.2f}%")

# ---
# Supplementary projections
# ---

# 5a. Quantitative supplementary: correlations with MCA axes
supp_quant_vars = ["eci_software", "gh_devs_per_10k", "gh_hill_q1_shannon"]
print("\n  Correlations of supplementary quantitative variables with MCA axes:")
print(f"  {'Variable':<25} {'r(Dim1)':>10} {'r(Dim2)':>10} {'r(Dim3)':>10}")
print(f"  {'-'*55}")
for var in supp_quant_vars:
    if var in df_active.columns:
        valid = df_active[var].notna()
        r1 = df_active.loc[valid, "mca_dim1"].corr(df_active.loc[valid, var])
        r2 = df_active.loc[valid, "mca_dim2"].corr(df_active.loc[valid, var])
        r3 = df_active.loc[valid, "mca_dim3"].corr(df_active.loc[valid, var]) if "mca_dim3" in df_active.columns else np.nan
        print(f"  {var:<25} {r1:>10.3f} {r2:>10.3f} {r3:>10.3f}")
    else:
        print(f"  {var:<25} (not available)")

# 5b. Categorical supplementary: mean coordinates by category
supp_cat_vars = ["region", "gh_has_devs"]
print("\n  Mean MCA coordinates by supplementary categorical variables:")
for var in supp_cat_vars:
    if var in df_active.columns:
        print(f"\n  --- {var} ---")
        grp = df_active.groupby(var)[["mca_dim1", "mca_dim2"]].agg(["mean", "count"])
        for cat in grp.index:
            d1 = grp.loc[cat, ("mca_dim1", "mean")]
            d2 = grp.loc[cat, ("mca_dim2", "mean")]
            n = int(grp.loc[cat, ("mca_dim1", "count")])
            print(f"    {str(cat):<20} Dim1={d1:>7.3f}  Dim2={d2:>7.3f}  (n={n})")

# ---
# Biplot (axes 1-2)
# ---

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot row points (departments) coloured by ECI gradient
has_eci = df_active["eci_software"].notna()
no_eci = ~has_eci

# Departments without ECI: grey
ax.scatter(
    df_active.loc[no_eci, "mca_dim1"],
    df_active.loc[no_eci, "mca_dim2"],
    c="lightgrey", s=15, alpha=0.4, zorder=1, label="No ECI"
)

# Departments with ECI: colour gradient
if has_eci.any():
    eci_vals = df_active.loc[has_eci, "eci_software"]
    sc = ax.scatter(
        df_active.loc[has_eci, "mca_dim1"],
        df_active.loc[has_eci, "mca_dim2"],
        c=eci_vals, cmap="RdYlGn", s=25, alpha=0.7, zorder=2,
        vmin=-2.5, vmax=2.5, edgecolors="none"
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("ECIsoftware (supplementary)", fontsize=10)

# Plot column points (modalities) as labelled markers
for idx in col_coords.index:
    d1 = col_coords.loc[idx, "Dim1"]
    d2 = col_coords.loc[idx, "Dim2"]
    label = str(idx)
    ax.plot(d1, d2, "k^", markersize=8, zorder=3)
    ax.annotate(
        label, (d1, d2), fontsize=7, fontweight="bold",
        xytext=(4, 4), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8),
        zorder=4
    )

# Supplementary categorical: region centroids
if "region" in df_active.columns:
    region_centroids = df_active.groupby("region")[["mca_dim1", "mca_dim2"]].mean()
    for reg in region_centroids.index:
        rx = region_centroids.loc[reg, "mca_dim1"]
        ry = region_centroids.loc[reg, "mca_dim2"]
        ax.plot(rx, ry, "s", markersize=12, color="navy", zorder=5)
        ax.annotate(
            reg, (rx, ry), fontsize=9, fontweight="bold", color="navy",
            xytext=(6, -6), textcoords="offset points", zorder=6
        )

# Axes
pct1 = pct_inertia[0]
pct2 = pct_inertia[1]
ax.set_xlabel(f"Dimension 1 ({pct1:.1f}% inertia)", fontsize=11)
ax.set_ylabel(f"Dimension 2 ({pct2:.1f}% inertia)", fontsize=11)
ax.set_title("MCA Biplot: Socioeconomic Space of Argentine Departments (N = {})".format(len(df_active)), fontsize=13)
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey", markersize=8, label="No ECI (supplementary)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=8, label="Active modalities"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="navy", markersize=8, label="Region centroids (suppl.)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
fig.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_02_mca_biplot.png"), dpi=300, bbox_inches="tight")
print("  Biplot saved: fig_mca_biplot.png")
plt.close()

# ---
# Eigenvalue scree plot
# ---

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Raw eigenvalues
ax1.bar(range(1, len(eigenvalues)+1), eigenvalues, color="steelblue", alpha=0.8)
ax1.axhline(threshold, color="red", linestyle="--", label=f"1/K = {threshold:.3f}")
ax1.set_xlabel("Axis")
ax1.set_ylabel("Eigenvalue")
ax1.set_title("Eigenvalue scree plot")
ax1.legend()

# Benzecri corrected
benz_nz = [b for b in benz_eigenvalues if b > 0]
ax2.bar(range(1, len(benz_nz)+1), benz_nz, color="darkorange", alpha=0.8)
benz_pct_nz = [b / benz_total * 100 for b in benz_nz]
ax2_twin = ax2.twinx()
ax2_twin.plot(range(1, len(benz_nz)+1), np.cumsum(benz_pct_nz), "ro-", markersize=6)
ax2_twin.set_ylabel("Cumulative % (corrected)")
ax2.set_xlabel("Axis")
ax2.set_ylabel("Corrected eigenvalue (Benzecri)")
ax2.set_title("Benzecri-corrected eigenvalues")

plt.tight_layout()
fig2.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_S2_mca_scree.png"), dpi=300, bbox_inches="tight")
print("  Scree plot saved: fig_mca_scree.png")
plt.close()

# ---
# Save MCA coordinates to database
# ---

# Prepare coordinates to save (up to n_retained dimensions)
dims_to_save = min(n_retained, 5)
save_cols = [f"mca_dim{i+1}" for i in range(dims_to_save)]

mca_save = df_active[["dpto5"] + save_cols].copy()
print(f"  Saving {dims_to_save} MCA dimensions for {len(mca_save)} departments")

# Update database: add MCA columns
with engine.connect() as conn:
    # Drop existing MCA columns if present
    for col in save_cols:
        try:
            conn.execute(text(f"ALTER TABLE art1.departamentos DROP COLUMN IF EXISTS {col}"))
        except Exception:
            pass
    conn.commit()

    # Add new columns
    for col in save_cols:
        conn.execute(text(f"ALTER TABLE art1.departamentos ADD COLUMN {col} DOUBLE PRECISION"))
    conn.commit()

    # Update values
    for _, row in mca_save.iterrows():
        set_clause = ", ".join([f"{col} = {row[col]:.8f}" for col in save_cols])
        conn.execute(text(
            f"UPDATE art1.departamentos SET {set_clause} WHERE dpto5 = '{row['dpto5']}'"
        ))
    conn.commit()

print(f"  MCA dimensions saved to art1.departamentos")

# Verify
verify = pd.read_sql(
    f"SELECT COUNT(*) as n, AVG(mca_dim1) as mean_d1, STDDEV(mca_dim1) as sd_d1 "
    f"FROM art1.departamentos WHERE mca_dim1 IS NOT NULL",
    engine
)
print(f"  Verification: {verify.iloc[0]['n']:.0f} departments with MCA coordinates")
print(f"    mean(Dim1) = {verify.iloc[0]['mean_d1']:.4f}, SD(Dim1) = {verify.iloc[0]['sd_d1']:.4f}")

# ---

print(f"\nDone. {len(df_active)} departments, {n_retained} axes retained, "
      f"{dims_to_save} dimensions saved to database.")
