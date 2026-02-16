"""
3x2 small-multiples map of Argentina showing the spatial distribution
of each departmental type (MCA-CAH clusters). Output: fig_04_cluster_maps.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sqlalchemy import create_engine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---

ENGINE_URL = "postgresql://postgres:postgres@localhost:5432/posadas"
OUTPUT_PATH = str(Path(__file__).resolve().parent.parent / "figures" / "fig_04_cluster_maps.png")

CLUSTER_COLOURS = {
    1: "#e41a1c",  # Peripheral-Deprived — red
    2: "#377eb8",  # Metropolitan-Core — blue
    3: "#4daf4a",  # Metropolitan-Diversified — green
    4: "#984ea3",  # Pampeana-Educated — purple
    5: "#ff7f00",  # Semi-Rural-Active — orange
    6: "#a65628",  # Intermediate-Urban — brown
}

CLUSTER_ORDER = [
    (2, "Metropolitan-Core"),
    (3, "Metropolitan-Diversified"),
    (4, "Pampeana-Educated"),
    (6, "Intermediate-Urban"),
    (1, "Peripheral-Deprived"),
    (5, "Semi-Rural-Active"),
]

BG_COLOUR = "#f0f0f0"       # light grey for non-highlighted departments
EDGE_COLOUR = "#cccccc"     # border colour for all departments
HIGHLIGHT_EDGE = "#333333"  # border colour for highlighted departments

# ---

engine = create_engine(ENGINE_URL)

# Load all department geometries
gdf = gpd.read_postgis(
    "SELECT redcode, geometry FROM public.departamentos_argentina",
    engine,
    geom_col="geometry",
)

# Dissolve CABA's 15 communes into a single unit (02000)
caba_mask = gdf["redcode"].str.startswith("02")
if caba_mask.any():
    caba_geom = gdf[caba_mask].geometry.union_all()
    caba_row = gpd.GeoDataFrame(
        [{"redcode": "02000", "geometry": caba_geom}],
        geometry="geometry",
        crs=gdf.crs,
    )
    gdf = pd.concat([gdf[~caba_mask], caba_row], ignore_index=True)

# Load cluster assignments (all 511 departments)
clusters = pd.read_sql(
    "SELECT dpto5, mca_cluster, mca_cluster_label FROM art1.departamentos "
    "WHERE mca_cluster IS NOT NULL",
    engine,
)

# Merge
gdf = gdf.merge(clusters, left_on="redcode", right_on="dpto5", how="left")

# Exclude Antarctica / islands far south for cleaner display
gdf = gdf[gdf.geometry.centroid.y > -56].copy()

# Get continental Argentina bounds (exclude far-flung islands)
minx, miny, maxx, maxy = gdf.total_bounds
# Tighten to continental
minx = max(minx, -74)
maxx = min(maxx, -53)
miny = max(miny, -56)
maxy = min(maxy, -21)

print(f"Total departments with geometry: {len(gdf)}")
print(f"Departments with cluster: {gdf['mca_cluster'].notna().sum()}")
print(f"Bounds: x=[{minx:.1f}, {maxx:.1f}], y=[{miny:.1f}, {maxy:.1f}]")

# ---

fig, axes = plt.subplots(3, 2, figsize=(8, 14))
fig.subplots_adjust(hspace=0.08, wspace=0.02, top=0.95, bottom=0.04,
                    left=0.02, right=0.98)

for idx, (cluster_id, cluster_label) in enumerate(CLUSTER_ORDER):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Draw all departments in background grey
    gdf.plot(
        ax=ax,
        color=BG_COLOUR,
        edgecolor=EDGE_COLOUR,
        linewidth=0.2,
    )

    # Highlight departments of this cluster type
    mask = gdf["mca_cluster"] == cluster_id
    n_type = mask.sum()
    colour = CLUSTER_COLOURS[cluster_id]

    gdf[mask].plot(
        ax=ax,
        color=colour,
        edgecolor=HIGHLIGHT_EDGE,
        linewidth=0.3,
        alpha=0.85,
    )

    # Set bounds
    ax.set_xlim(minx - 0.5, maxx + 0.5)
    ax.set_ylim(miny - 0.5, maxy + 0.5)

    # Title with count
    ax.set_title(
        f"{cluster_label}\n(n = {n_type})",
        fontsize=11,
        fontweight="bold",
        color=colour,
        pad=4,
    )

    # Clean up axes
    ax.set_aspect("equal")
    ax.axis("off")

# Save
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved to: {OUTPUT_PATH}")
plt.close(fig)
