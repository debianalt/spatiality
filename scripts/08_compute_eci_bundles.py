"""
Bundle robustness check: recompute ECI using Juhasz et al. (2026) language-bundle
taxonomy. Programming languages are aggregated into 38 named clusters before the
ECI eigenvalue decomposition; the result (eci_bundle) is compared with eci_software
(individual-language ECI) to verify robustness.

Reference: Juhasz, S., Wachs, J., Kamiński, J., & Hidalgo, C. A. (2026).
  The software complexity of nations. Research Policy, 55(3), 105422.
  Replication data: github.com/sandorjuhasz/eci_software
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sqlalchemy import create_engine, text
from scipy.linalg import eig
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_URL = "postgresql://postgres:postgres@localhost:5432/posadas"
engine = create_engine(ENGINE_URL)

MAPPING_URL = (
    "https://raw.githubusercontent.com/sandorjuhasz/eci_software/"
    "main/data/outputs/language_to_cluster_mapping.csv"
)
MAPPING_CACHE = REPO_ROOT / "data" / "juhasz_language_clusters.csv"


def load_bundle_mapping() -> pd.DataFrame:
    """Return Juhasz et al. mapping as a DataFrame with columns [language, bundle]."""
    if MAPPING_CACHE.exists():
        print(f"Loading cached bundle mapping from {MAPPING_CACHE}")
        raw = pd.read_csv(MAPPING_CACHE)
    else:
        print(f"Downloading bundle mapping from GitHub...")
        r = requests.get(MAPPING_URL, timeout=30)
        r.raise_for_status()
        MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MAPPING_CACHE.write_bytes(r.content)
        raw = pd.read_csv(MAPPING_CACHE)

    print(f"  Loaded {len(raw)} rows. Columns: {list(raw.columns)}")
    print(raw.head(3).to_string(index=False))

    # Normalise column names: lowercase + strip, replace spaces with underscores
    cols_norm = {c: c.lower().strip().replace(" ", "_") for c in raw.columns}
    raw = raw.rename(columns=cols_norm)

    # Identify the bundle-name column (prefer a string/name column over numeric id)
    name_col = None
    for candidate in ["cluster_name", "cluster", "bundle", "bundle_name",
                      "cluster_label", "group_name"]:
        if candidate in raw.columns:
            name_col = candidate
            break
    if name_col is None:
        # Fall back: take the first non-'language', non-numeric column
        for c in raw.columns:
            if c != "language" and raw[c].dtype == object:
                name_col = c
                break
    if name_col is None:
        # Last resort: use cluster_id as bundle identifier
        name_col = [c for c in raw.columns if c != "language"][0]
        print(f"  Warning: using '{name_col}' as bundle identifier (no name column found)")

    mapping = raw[["language", name_col]].copy()
    mapping.columns = ["language", "bundle"]
    mapping["language"] = mapping["language"].str.strip()
    mapping["bundle"] = mapping["bundle"].astype(str).str.strip()

    print(f"  Distinct bundles: {mapping['bundle'].nunique()}")
    return mapping


# ── 1. Load Juhasz et al. bundle mapping ──────────────────────────────────────
bundle_map = load_bundle_mapping()
lang_to_bundle = dict(zip(bundle_map["language"], bundle_map["bundle"]))

# ── 2. Load repository data (identical pipeline to 01_compute_eci.py) ─────────
FOREIGN_USERS = pd.read_csv(
    REPO_ROOT / "audit" / "audit_04_foreign_users.csv",
    usecols=["username"],
)["username"].tolist()
print(f"\nForeign-user exclusion list: {len(FOREIGN_USERS)} users")

query = text("""
SELECT LEFT(redcode, 5) AS dpto5,
       primary_language,
       COUNT(*)          AS repos
FROM   github_argentina.repos
WHERE  primary_language IS NOT NULL
  AND  primary_language != ''
  AND  username NOT IN (SELECT unnest(:foreign_users))
GROUP  BY 1, 2
""")
df_raw = pd.read_sql(query, engine, params={"foreign_users": FOREIGN_USERS})

EXCLUDE_CODES = {"94021", "94028"}

CORDOBA_CORRECTIONS = {
    "14112": "14119", "14119": "14126", "14126": "14133",
    "14133": "14140", "14140": "14147", "14154": "14161",
    "14175": "14182", "14182": "14112",
}
MAPPING_CODES = {
    "06217": "06218", "06466": "06218",
    "94007": "94008", "94014": "94015", "94011": "94015",
    **CORDOBA_CORRECTIONS,
}

df_raw["dpto5"] = df_raw["dpto5"].apply(
    lambda x: "02000" if x.startswith("02") else MAPPING_CODES.get(x, x)
)
df_raw = df_raw[~df_raw["dpto5"].isin(EXCLUDE_CODES)].copy()

df_agg = (
    df_raw
    .groupby(["dpto5", "primary_language"], as_index=False)["repos"]
    .sum()
)

geo_totals = pd.read_sql(
    "SELECT LEFT(redcode, 5) AS dpto5, total_repos::int AS geo_repos "
    "FROM github_argentina.departamentos_geo",
    engine,
)
zero_geo = set(geo_totals.loc[geo_totals["geo_repos"] == 0, "dpto5"])
spurious = set(df_agg["dpto5"].unique()) & zero_geo
if spurious:
    df_agg = df_agg[~df_agg["dpto5"].isin(spurious)].copy()

# ── 3. Map languages → bundles ─────────────────────────────────────────────────
argentine_langs = sorted(df_agg["primary_language"].unique())
n_arg_langs = len(argentine_langs)

mapped = [l for l in argentine_langs if l in lang_to_bundle]
unmapped = [l for l in argentine_langs if l not in lang_to_bundle]

print(f"\n{'='*70}")
print(f"LANGUAGE -> BUNDLE COVERAGE")
print(f"{'='*70}")
print(f"Argentine languages in filtered dataset:  {n_arg_langs}")
print(f"  Mapped to Juhasz bundle:                {len(mapped)}")
print(f"  Unmapped (kept as singleton bundles):   {len(unmapped)}")
if unmapped:
    print(f"  Unmapped languages: {unmapped}")

repos_mapped = df_agg.loc[df_agg["primary_language"].isin(mapped), "repos"].sum()
repos_total  = df_agg["repos"].sum()
print(f"\nRepositories in mapped languages: {repos_mapped:,} "
      f"({100*repos_mapped/repos_total:.1f}% of total)")

# Assign bundle: mapped → Juhasz bundle; unmapped → language name itself
df_agg["bundle"] = df_agg["primary_language"].map(lang_to_bundle).fillna(
    df_agg["primary_language"]
)

# ── 4. Apply dept/bundle thresholds ───────────────────────────────────────────
df_bundle_agg = (
    df_agg
    .groupby(["dpto5", "bundle"], as_index=False)["repos"]
    .sum()
)

dept_totals = df_bundle_agg.groupby("dpto5")["repos"].sum()
valid_depts = dept_totals[dept_totals >= 10].index

bundle_totals = df_bundle_agg.groupby("bundle")["repos"].sum()
valid_bundles = bundle_totals[bundle_totals >= 30].index

print(f"\nDepartments with >=10 repos: {len(valid_depts)}")
print(f"Bundles with >=30 repos nationally: {len(valid_bundles)} "
      f"(from {bundle_totals.shape[0]} total bundles)")

df_filt = df_bundle_agg[
    df_bundle_agg["dpto5"].isin(valid_depts) &
    df_bundle_agg["bundle"].isin(valid_bundles)
].copy()

M = df_filt.pivot_table(
    index="dpto5", columns="bundle", values="repos", fill_value=0
)
print(f"Bundle matrix M: {M.shape[0]} departments x {M.shape[1]} bundles")

# ── 5. ECI eigenvalue decomposition (identical to 01_compute_eci.py) ──────────
M_vals = M.values.astype(float)
row_sums = M_vals.sum(axis=1, keepdims=True)
col_sums = M_vals.sum(axis=0, keepdims=True)
total = M_vals.sum()

RCA = (M_vals / row_sums) / (col_sums / total)
A = (RCA >= 1).astype(float)

print(f"Non-zero RCA entries: {int(A.sum())} / {A.size} ({100*A.sum()/A.size:.1f}%)")

diversity = A.sum(axis=1)
ubiquity  = A.sum(axis=0)
diversity[diversity == 0] = 1e-10
ubiquity[ubiquity == 0]   = 1e-10

D_inv = np.diag(1.0 / diversity)
U_inv = np.diag(1.0 / ubiquity)
M_tilde = D_inv @ A @ U_inv @ A.T

eigenvalues, eigenvectors = eig(M_tilde)
idx_sorted = np.argsort(-eigenvalues.real)
eigenvalues  = eigenvalues[idx_sorted]
eigenvectors = eigenvectors[:, idx_sorted]

eci_raw = eigenvectors[:, 1].real
eci_std = (eci_raw - eci_raw.mean()) / eci_raw.std()

if np.corrcoef(eci_std, diversity)[0, 1] < 0:
    eci_std = -eci_std
    print("ECI_bundle sign flipped to align with diversity.")

eci_bundle_df = pd.DataFrame({
    "dpto5": M.index,
    "eci_bundle": eci_std,
})

# ── 6. Correlation with individual-language ECI ───────────────────────────────
eci_individual = pd.read_sql(
    "SELECT dpto5, eci_software FROM art1.departamentos WHERE eci_software IS NOT NULL",
    engine,
)

merged = eci_bundle_df.merge(eci_individual, on="dpto5", how="inner")
n_shared = len(merged)
pearson_r  = merged["eci_bundle"].corr(merged["eci_software"])
spearman_r = spearmanr(merged["eci_bundle"], merged["eci_software"]).statistic

print(f"\n{'='*70}")
print(f"ROBUSTNESS CHECK: ECI_bundle vs ECI_software (individual languages)")
print(f"{'='*70}")
print(f"  Departments with both measures: {n_shared}")
print(f"  Pearson r:   {pearson_r:+.4f}")
print(f"  Spearman rho: {spearman_r:+.4f}")

if abs(spearman_r) >= 0.75:
    print(f"  [STRONG] Concordance >= 0.75: bundle approach corroborates individual-language ECI.")
elif abs(spearman_r) >= 0.50:
    print(f"  [MODERATE] Concordance 0.50-0.75: results broadly consistent with caveats.")
else:
    print(f"  [WEAK] Concordance < 0.50: bundle aggregation alters rankings substantially.")

# ── 7. Key department rankings comparison ─────────────────────────────────────
dept_names = pd.read_sql(
    "SELECT dpto5, departamento, provincia FROM art1.departamentos", engine
)
merged_named = merged.merge(dept_names, on="dpto5", how="left")

merged_named["rank_individual"] = merged_named["eci_software"].rank(
    ascending=False, method="min"
).astype(int)
merged_named["rank_bundle"] = merged_named["eci_bundle"].rank(
    ascending=False, method="min"
).astype(int)
merged_named["rank_diff"] = (
    merged_named["rank_individual"] - merged_named["rank_bundle"]
)

key_depts = ["02000", "14098", "58091", "62049", "18021", "06270"]  # CABA, Córdoba Cap, Bariloche, Neuquén Cap, Corrientes Cap, La Plata
key_rows = merged_named[merged_named["dpto5"].isin(key_depts)].sort_values("rank_individual")

print(f"\n--- Rank comparison for selected departments (out of {n_shared}) ---")
print(f"  {'dpto5':<7} {'Departamento':<30} {'ECI_ind':>8} {'ECI_bun':>8} "
      f"{'Rank_ind':>8} {'Rank_bun':>8} {'dRank':>6}")
for _, r in key_rows.iterrows():
    dname = str(r["departamento"])[:28] if r["departamento"] else "N/A"
    print(f"  {r['dpto5']:<7} {dname:<30} {r['eci_software']:+8.4f} "
          f"{r['eci_bundle']:+8.4f} {int(r['rank_individual']):>8} "
          f"{int(r['rank_bundle']):>8} {int(r['rank_diff']):>+6}")

print(f"\n--- Top 15 departments by ECI_bundle ---")
top15 = merged_named.nlargest(15, "eci_bundle")
for _, r in top15.iterrows():
    dname = str(r["departamento"])[:28] if r["departamento"] else "N/A"
    print(f"  {r['dpto5']}  {dname:<30}  ECI_bun={r['eci_bundle']:+.4f}  "
          f"ECI_ind={r['eci_software']:+.4f}  dRank={int(r['rank_diff']):+d}")

print(f"\n--- Largest rank changes (|dRank| > 10) ---")
large_moves = merged_named[abs(merged_named["rank_diff"]) > 10].sort_values(
    "rank_diff", key=abs, ascending=False
)
for _, r in large_moves.head(20).iterrows():
    dname = str(r["departamento"])[:28] if r["departamento"] else "N/A"
    print(f"  {r['dpto5']}  {dname:<30}  ECI_ind={r['eci_software']:+.4f}  "
          f"ECI_bun={r['eci_bundle']:+.4f}  dRank={int(r['rank_diff']):+d}")

# ── 8. Bundle-level PCI ────────────────────────────────────────────────────────
n_l = M_vals.shape[1]
M_tilde_lang = U_inv @ A.T @ D_inv @ A
eigenvalues_l, eigenvectors_l = eig(M_tilde_lang)
idx_sorted_l = np.argsort(-eigenvalues_l.real)
pci_raw = eigenvectors_l[:, 1].real
pci_std = (pci_raw - pci_raw.mean()) / pci_raw.std()

if np.corrcoef(pci_std, ubiquity)[0, 1] > 0:
    pci_std = -pci_std

pci_bundle_df = pd.DataFrame({
    "bundle": M.columns,
    "pci_bundle": pci_std,
    "ubiquity": ubiquity,
}).sort_values("pci_bundle", ascending=False)

print(f"\n--- Top 10 bundles by PCI_bundle ---")
for _, r in pci_bundle_df.head(10).iterrows():
    print(f"  {str(r['bundle']):<40}  PCI={r['pci_bundle']:+.4f}  ubiquity={int(r['ubiquity'])}")

print(f"\n--- Bottom 10 bundles by PCI_bundle ---")
for _, r in pci_bundle_df.tail(10).iterrows():
    print(f"  {str(r['bundle']):<40}  PCI={r['pci_bundle']:+.4f}  ubiquity={int(r['ubiquity'])}")

# ── 9. Write eci_bundle to PostgreSQL ─────────────────────────────────────────
with engine.begin() as conn:
    conn.execute(text(
        "DO $$ BEGIN "
        "IF NOT EXISTS ("
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'art1' AND table_name = 'departamentos' "
        "AND column_name = 'eci_bundle'"
        ") THEN "
        "ALTER TABLE art1.departamentos ADD COLUMN eci_bundle DOUBLE PRECISION; "
        "END IF; END $$;"
    ))
    conn.execute(text(
        "UPDATE art1.departamentos SET eci_bundle = NULL"
    ))
    update_sql = text(
        "UPDATE art1.departamentos "
        "SET eci_bundle = :eci_bundle "
        "WHERE dpto5 = :dpto5"
    )
    rows_updated = 0
    for _, row in eci_bundle_df.iterrows():
        result = conn.execute(update_sql, {
            "eci_bundle": float(row["eci_bundle"]),
            "dpto5": row["dpto5"],
        })
        rows_updated += result.rowcount

print(f"\nPostgreSQL: {rows_updated} rows updated in art1.departamentos (eci_bundle).")

# ── 10. Export comparison table for supplementary material ────────────────────
supp_table = merged_named[[
    "dpto5", "departamento", "provincia",
    "eci_software", "rank_individual",
    "eci_bundle",   "rank_bundle",
    "rank_diff",
]].sort_values("rank_individual").copy()

supp_table.columns = [
    "dpto5", "Departamento", "Provincia",
    "ECI_individual", "Rank_individual",
    "ECI_bundle",     "Rank_bundle",
    "Rank_change",
]

out_path = REPO_ROOT / "data" / "table_s_bundle_robustness.csv"
supp_table.to_csv(out_path, index=False, float_format="%.4f")
print(f"Supplementary table exported to: {out_path}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Argentine languages: {n_arg_langs} -> "
      f"{len(mapped)} mapped to Juhasz bundles ({len(unmapped)} singletons)")
print(f"  Bundles retained (>=30 repos nationally): {len(valid_bundles)}")
print(f"  Matrix: {M.shape[0]} departments x {M.shape[1]} bundles")
print(f"  Pearson r  (ECI_bundle vs ECI_individual): {pearson_r:+.4f}")
print(f"  Spearman rho (ECI_bundle vs ECI_individual): {spearman_r:+.4f}")
print("\nDone.")
