"""
Table S5: Pearson correlation matrix for MCA active variables and ECIsoftware.
N=511 for MCA variables; N=224 for correlations involving ECI.
Output: supplementary/table_S5_correlation_matrix.csv + formatted text.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sqlalchemy
from scipy import stats

engine = sqlalchemy.create_engine("postgresql://postgres:postgres@localhost/posadas")
df = pd.read_sql("SELECT * FROM art1.departamentos", engine)

# Variable definitions
VARS = {
    "sec_educ":   "Secondary education",
    "uni_educ":   "University education",
    "pc_own":     "Computer ownership",
    "rad_viirs":  "Nighttime radiance",
    "emp_rate":   "Employment rate",
    "nbi":        "Unsatisfied basic needs",
    "overcrwd":   "Overcrowding",
    "ln_pop":     "Population (log)",
    "eci_software": "ECIsoftware",
}

# Map to actual column names in art1.departamentos
COL_MAP = {
    "sec_educ":     "pct_jefe_sec_2010",
    "uni_educ":     "pct_jefe_uni_2010",
    "pc_own":       "pct_pc_2010",
    "rad_viirs":    "rad_2014",
    "emp_rate":     "tasa_empleo_2010",
    "nbi":          "pct_nbi_2010",
    "overcrwd":     "pct_hacinam_2010",
    "ln_pop":       None,            # computed below from pob_2010
    "eci_software": "eci_software",
}

# Build subset — ln_pop computed
raw_cols = [v for v in COL_MAP.values() if v is not None]
sub_raw = df[raw_cols + ["pob_2010"]].copy()
sub_raw["ln_pop"] = np.log(sub_raw["pob_2010"])

short_keys = list(VARS.keys())
labels = list(VARS.values())

sub = pd.DataFrame()
for k, v in COL_MAP.items():
    if v is None:
        sub[k] = sub_raw["ln_pop"]
    else:
        sub[k] = sub_raw[v]

# Correlation matrix with p-values (pairwise, N varies due to ECI missingness)
n = len(short_keys)
r_mat = np.full((n, n), np.nan)
p_mat = np.full((n, n), np.nan)
n_mat = np.full((n, n), np.nan, dtype=float)

for i, vi in enumerate(short_keys):
    for j, vj in enumerate(short_keys):
        mask = sub[[vi, vj]].notna().all(axis=1)
        n_ij = mask.sum()
        if n_ij > 2:
            r, p = stats.pearsonr(sub.loc[mask, vi], sub.loc[mask, vj])
            r_mat[i, j] = r
            p_mat[i, j] = p
            n_mat[i, j] = n_ij

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# Build formatted matrix (lower triangle only, diagonal = 1)
rows = []
for i, label_i in enumerate(labels):
    row = {"Variable": label_i}
    for j, key_j in enumerate(short_keys):
        if j > i:
            row[key_j] = ""
        elif j == i:
            row[key_j] = "1"
        else:
            r = r_mat[i, j]
            p = p_mat[i, j]
            row[key_j] = f"{r:.2f}{sig_stars(p)}"
    rows.append(row)

result = pd.DataFrame(rows)
result.index = range(1, n + 1)

out_dir = Path(__file__).parent.parent / "supplementary"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "table_S5_correlation_matrix.csv"
result.to_csv(out_path, index=True)

print("Table S5: Pearson correlation matrix")
print(f"N (MCA variables) = {int(n_mat[0,1])}")
print(f"N (ECI pairs)     = {int(n_mat[0, -1])}")
print(result.to_string())
print(f"\nSaved to {out_path}")
print("\n*** p<0.001  ** p<0.01  * p<0.05")
