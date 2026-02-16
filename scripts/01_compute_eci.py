"""
Compute ECI_software at department level (Hidalgo & Hausmann 2009).
Products = programming languages; locations = departments.
Eigenvalue decomposition on the normalised adjacency matrix of the
bipartite dept-language network; results written to art1.departamentos.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from scipy.linalg import eig

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_URL = "postgresql://postgres:postgres@localhost:5432/posadas"
engine = create_engine(ENGINE_URL)

FOREIGN_USERS = pd.read_csv(
    REPO_ROOT / "audit" / "audit_04_foreign_users.csv",
    usecols=["username"],
)["username"].tolist()
print(f"Foreign-user exclusion list: {len(FOREIGN_USERS)} users")

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
print(f"Raw query: {len(df_raw):,} rows, "
      f"{df_raw['dpto5'].nunique()} distinct dept codes, "
      f"{df_raw['primary_language'].nunique()} distinct languages, "
      f"{df_raw['repos'].sum():,} total repos with language data.")

EXCLUDE_CODES = {"94021", "94028"}

# Córdoba off-by-one geocoding correction (verified against spatial join)
CORDOBA_CORRECTIONS = {
    "14112": "14119",   # Río Seco → Río Segundo
    "14119": "14126",   # Río Segundo → San Alberto
    "14126": "14133",   # San Alberto → San Javier
    "14133": "14140",   # San Javier → San Justo
    "14140": "14147",   # San Justo → Santa María
    "14154": "14161",   # Sobremonte → Tercero Arriba
    "14175": "14182",   # Tulumba → Unión
    "14182": "14112",   # Unión → Río Seco
}

MAPPING = {
    "06217": "06218",
    "06466": "06218",
    "94007": "94008",
    "94014": "94015",
    "94011": "94015",
    **CORDOBA_CORRECTIONS,
}

df_raw["dpto5"] = df_raw["dpto5"].apply(
    lambda x: "02000" if x.startswith("02") else MAPPING.get(x, x)
)

df_raw = df_raw[~df_raw["dpto5"].isin(EXCLUDE_CODES)].copy()

df_agg = (
    df_raw
    .groupby(["dpto5", "primary_language"], as_index=False)["repos"]
    .sum()
)
print(f"After mapping/aggregation: {len(df_agg):,} rows, "
      f"{df_agg['dpto5'].nunique()} departments, "
      f"{df_agg['primary_language'].nunique()} languages.")

# Cross-validate against spatially-joined departamentos_geo
geo_totals = pd.read_sql(
    "SELECT LEFT(redcode, 5) AS dpto5, total_repos::int AS geo_repos "
    "FROM github_argentina.departamentos_geo",
    engine,
)
zero_geo = set(geo_totals.loc[geo_totals["geo_repos"] == 0, "dpto5"])
spurious = set(df_agg["dpto5"].unique()) & zero_geo
if spurious:
    df_agg = df_agg[~df_agg["dpto5"].isin(spurious)].copy()
    print(f"Cross-validation: excluded {len(spurious)} departments with 0 repos "
          f"in departamentos_geo (spurious geocoding): {sorted(spurious)}")

dept_totals = df_agg.groupby("dpto5")["repos"].sum()
valid_depts = dept_totals[dept_totals >= 10].index
print(f"Departments with >= 10 repos: {len(valid_depts)} "
      f"(excluded {df_agg['dpto5'].nunique() - len(valid_depts)})")

lang_totals = df_agg.groupby("primary_language")["repos"].sum()
valid_langs = lang_totals[lang_totals >= 30].index
print(f"Languages with >= 30 repos nationally: {len(valid_langs)} "
      f"(excluded {df_agg['primary_language'].nunique() - len(valid_langs)})")

df_filt = df_agg[
    df_agg["dpto5"].isin(valid_depts) & df_agg["primary_language"].isin(valid_langs)
].copy()

M = df_filt.pivot_table(
    index="dpto5", columns="primary_language", values="repos", fill_value=0
)
print(f"\nFiltered matrix M: {M.shape[0]} departments x {M.shape[1]} languages")

M_vals = M.values.astype(float)
row_sums = M_vals.sum(axis=1, keepdims=True)
col_sums = M_vals.sum(axis=0, keepdims=True)
total = M_vals.sum()

RCA = (M_vals / row_sums) / (col_sums / total)
A = (RCA >= 1).astype(float)

print(f"RCA matrix computed. Non-zero entries in A (RCA>=1): "
      f"{int(A.sum())} out of {A.size} ({100*A.sum()/A.size:.1f}%)")

diversity = A.sum(axis=1)
ubiquity = A.sum(axis=0)

diversity[diversity == 0] = 1e-10
ubiquity[ubiquity == 0] = 1e-10

n_d = M_vals.shape[0]

D_inv = np.diag(1.0 / diversity)
U_inv = np.diag(1.0 / ubiquity)

M_tilde = D_inv @ A @ U_inv @ A.T

eigenvalues, eigenvectors = eig(M_tilde)

idx_sorted = np.argsort(-eigenvalues.real)
eigenvalues = eigenvalues[idx_sorted]
eigenvectors = eigenvectors[:, idx_sorted]

print(f"\nTop 5 eigenvalues (real part): {eigenvalues[:5].real}")

eci_raw = eigenvectors[:, 1].real
eci_std = (eci_raw - eci_raw.mean()) / eci_raw.std()

if np.corrcoef(eci_std, diversity)[0, 1] < 0:
    eci_std = -eci_std
    print("ECI sign flipped to align with diversity.")

diversity_series = pd.Series(A.sum(axis=1), index=M.index, name="eci_diversity")

avg_ubiq = np.zeros(n_d)
for d in range(n_d):
    mask = A[d, :] == 1
    if mask.any():
        avg_ubiq[d] = ubiquity[mask].mean()
    else:
        avg_ubiq[d] = np.nan
avg_ubiq_series = pd.Series(avg_ubiq, index=M.index, name="eci_avg_ubiquity")

n_l = M_vals.shape[1]
M_tilde_lang = U_inv @ A.T @ D_inv @ A

eigenvalues_l, eigenvectors_l = eig(M_tilde_lang)
idx_sorted_l = np.argsort(-eigenvalues_l.real)
eigenvalues_l = eigenvalues_l[idx_sorted_l]
eigenvectors_l = eigenvectors_l[:, idx_sorted_l]

pci_raw = eigenvectors_l[:, 1].real
pci_std = (pci_raw - pci_raw.mean()) / pci_raw.std()

if np.corrcoef(pci_std, ubiquity)[0, 1] > 0:
    pci_std = -pci_std
    print("PCI sign flipped to align with lower ubiquity = higher complexity.")

pci_series = pd.Series(pci_std, index=M.columns, name="pci")

eci_df = pd.DataFrame({
    "dpto5": M.index,
    "eci_software": eci_std,
    "eci_diversity": diversity_series.values,
    "eci_avg_ubiquity": avg_ubiq_series.values,
})

with engine.begin() as conn:
    for col, dtype in [("eci_software", "DOUBLE PRECISION"),
                       ("eci_diversity", "INTEGER"),
                       ("eci_avg_ubiquity", "DOUBLE PRECISION")]:
        conn.execute(text(
            "DO $$ BEGIN "
            "IF NOT EXISTS ("
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'art1' "
            "AND table_name = 'departamentos' "
            f"AND column_name = '{col}'"
            ") THEN "
            f"ALTER TABLE art1.departamentos ADD COLUMN {col} {dtype}; "
            "END IF; END $$;"
        ))

    conn.execute(text(
        "UPDATE art1.departamentos "
        "SET eci_software = NULL, "
        "    eci_diversity = NULL, "
        "    eci_avg_ubiquity = NULL"
    ))

    update_sql = text(
        "UPDATE art1.departamentos "
        "SET eci_software = :eci_software, "
        "    eci_diversity = :eci_diversity, "
        "    eci_avg_ubiquity = :eci_avg_ubiquity "
        "WHERE dpto5 = :dpto5"
    )
    rows_updated = 0
    for _, row in eci_df.iterrows():
        ubiq_val = float(row["eci_avg_ubiquity"]) if not np.isnan(row["eci_avg_ubiquity"]) else None
        result = conn.execute(update_sql, {
            "eci_software": float(row["eci_software"]),
            "eci_diversity": int(row["eci_diversity"]),
            "eci_avg_ubiquity": ubiq_val,
            "dpto5": row["dpto5"],
        })
        rows_updated += result.rowcount

    print(f"\nPostgreSQL update: {rows_updated} rows updated in art1.departamentos.")

print("\n" + "=" * 70)
print("SUMMARY STATISTICS -- ECI_software")
print("=" * 70)
print(f"  N departments with ECI: {len(eci_df)}")
print(f"  Min:    {eci_df['eci_software'].min():.4f}")
print(f"  Max:    {eci_df['eci_software'].max():.4f}")
print(f"  Mean:   {eci_df['eci_software'].mean():.4f}")
print(f"  Median: {eci_df['eci_software'].median():.4f}")
print(f"  Std:    {eci_df['eci_software'].std():.4f}")

dept_names = pd.read_sql(
    "SELECT dpto5, departamento FROM art1.departamentos", engine
)
eci_named = eci_df.merge(dept_names, on="dpto5", how="left")

print("\n--- Top 20 departments by ECI_software ---")
top20 = eci_named.nlargest(20, "eci_software")
for rank, (i, r) in enumerate(top20.iterrows(), 1):
    dname = str(r["departamento"]) if r["departamento"] else "N/A"
    print(f"  {rank:>2d}. {r['dpto5']}  {dname:<30s}  ECI={r['eci_software']:+.4f}  "
          f"div={int(r['eci_diversity'])}  avg_ubiq={r['eci_avg_ubiquity']:.1f}")

print("\n--- Bottom 10 departments by ECI_software ---")
bot10 = eci_named.nsmallest(10, "eci_software")
for i, r in bot10.iterrows():
    dname = str(r["departamento"]) if r["departamento"] else "N/A"
    print(f"  {r['dpto5']}  {dname:<30s}  ECI={r['eci_software']:+.4f}  "
          f"div={int(r['eci_diversity'])}  avg_ubiq={r['eci_avg_ubiquity']:.1f}")

corr_vars = ["gh_total_developers", "gh_devs_per_10k", "pob_2022", "pct_nbi_2022"]
corr_query = (
    "SELECT dpto5, eci_software, "
    + ", ".join(corr_vars)
    + " FROM art1.departamentos WHERE eci_software IS NOT NULL"
)
df_corr = pd.read_sql(corr_query, engine)

print("\n--- Correlations of ECI_software with key variables ---")
for var in corr_vars:
    subset = df_corr[["eci_software", var]].dropna()
    if len(subset) > 2:
        r_val = subset["eci_software"].corr(subset[var])
        print(f"  ECI_software vs {var:<25s}: r = {r_val:+.4f}  (n = {len(subset)})")
    else:
        print(f"  ECI_software vs {var:<25s}: insufficient data")

print("\n--- Top 10 languages by PCI (Product Complexity Index) ---")
pci_df_out = pd.DataFrame({"language": M.columns, "pci": pci_std})
pci_df_out = pci_df_out.sort_values("pci", ascending=False)
for i, r in pci_df_out.head(10).iterrows():
    ubiq_val = ubiquity[list(M.columns).index(r["language"])]
    print(f"  {r['language']:<25s}  PCI={r['pci']:+.4f}  ubiquity={int(ubiq_val)}")

print("\n--- Bottom 10 languages by PCI ---")
for i, r in pci_df_out.tail(10).iterrows():
    ubiq_val = ubiquity[list(M.columns).index(r["language"])]
    print(f"  {r['language']:<25s}  PCI={r['pci']:+.4f}  ubiquity={int(ubiq_val)}")

print("\nDone.")
