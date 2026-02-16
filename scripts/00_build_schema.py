"""
Build art1 schema: analysis-ready tables for the article.
Handles CABA aggregation, Chascomus/Lezama merge, TdF mapping.
Creates art1.departamentos (511 depts) and art1.secuencias (dpto x year).
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DB = "postgresql://postgres:postgres@localhost:5432/posadas"
engine = create_engine(DB)

# Department code mapping
def map_dpto(code):
    """Harmonise department codes across all sources."""
    code = str(code).zfill(5)
    # CABA: all communes → 02000
    if code[:2] == '02':
        return '02000'
    # Chascomús 2010 code → 2022 code
    if code == '06217':
        return '06218'
    # Lezama → merge into Chascomús
    if code == '06466':
        return '06218'
    # Río Grande 2010 → 2022 code
    if code == '94007':
        return '94008'
    # Ushuaia 2010 → 2022 code
    if code == '94014':
        return '94015'
    # Tolhuin → merge into Ushuaia
    if code == '94011':
        return '94015'
    # Antarctic territories → exclude
    if code in ('94021', '94028'):
        return None
    return code

# Region assignment
REGION_MAP = {
    '02': 'CABA', '06': 'Pampeana', '14': 'Pampeana', '30': 'Pampeana',
    '42': 'Pampeana', '82': 'Pampeana',
    '10': 'NOA', '38': 'NOA', '46': 'NOA', '66': 'NOA', '86': 'NOA', '90': 'NOA',
    '18': 'NEA', '22': 'NEA', '34': 'NEA', '54': 'NEA',
    '50': 'Cuyo', '70': 'Cuyo', '74': 'Cuyo',
    '26': 'Patagonia', '58': 'Patagonia', '62': 'Patagonia',
    '78': 'Patagonia', '94': 'Patagonia',
}

def get_region(dpto5):
    return REGION_MAP.get(str(dpto5)[:2], 'Unknown')

# Helper: read SQL, map codes, aggregate
def read_and_map(sql, dpto_col='dpto5'):
    """Read SQL, apply department mapping, return DataFrame."""
    df = pd.read_sql(sql, engine)
    df['dpto5'] = df[dpto_col].apply(map_dpto)
    df = df[df['dpto5'].notna()].copy()
    return df

# ---
# Census 2010
# ---
print("Loading Census 2010...", flush=True)
c10 = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        MAX(prov) as prov_2010, MAX(dpto) as dpto_2010,
        SUM(radios_pob) as pob_2010,
        SUM(h_total) as hog_2010,
        SUM(h_nbi) as h_nbi_2010,
        SUM(h_computad) as h_pc_2010,
        SUM(h_celular) as h_cel_2010,
        SUM(h_telefono) as h_tel_2010,
        SUM(h_heladera) as h_heladera_2010,
        SUM(h_cloaca) as h_cloaca_2010,
        SUM(h_agua_red) as h_agua_red_2010,
        SUM(h_combusti) as h_gas_2010,
        SUM(h_propieta) as h_prop_2010,
        SUM(h_inquilin) as h_inq_2010,
        SUM(h_jefe_muj) as h_jefa_muj_2010,
        SUM(h_jefe_lim) as h_jefe_lim_2010,
        SUM(h_jefe_sec) as h_jefe_sec_2010,
        SUM(h_jefe_ter) as h_jefe_ter_2010,
        SUM(h_jefe_uni) as h_jefe_uni_2010,
        SUM(h_domestic) as h_domestic_2010,
        SUM(h_hacinami) as h_hacinam_2010,
        SUM(h_cadena) as h_cadena_2010,
        SUM(radios_sup) as area_km2_2010
    FROM censo_2010.nbi_2010
    GROUP BY SUBSTRING(redcode,1,5)
""")
# Aggregate mapped departments
agg_cols_c10 = [c for c in c10.columns if c not in ('dpto5','prov_2010','dpto_2010')]
c10_agg = c10.groupby('dpto5').agg(
    prov_2010=('prov_2010', 'first'),
    dpto_2010=('dpto_2010', 'first'),
    **{c: (c, 'sum') for c in agg_cols_c10}
).reset_index()
# Compute percentages
for var, num, den in [
    ('pct_nbi_2010', 'h_nbi_2010', 'hog_2010'),
    ('pct_pc_2010', 'h_pc_2010', 'hog_2010'),
    ('pct_cel_2010', 'h_cel_2010', 'hog_2010'),
    ('pct_cloaca_2010', 'h_cloaca_2010', 'hog_2010'),
    ('pct_agua_red_2010', 'h_agua_red_2010', 'hog_2010'),
    ('pct_gas_2010', 'h_gas_2010', 'hog_2010'),
    ('pct_prop_2010', 'h_prop_2010', 'hog_2010'),
    ('pct_jefa_muj_2010', 'h_jefa_muj_2010', 'hog_2010'),
    ('pct_jefe_lim_2010', 'h_jefe_lim_2010', 'hog_2010'),
    ('pct_jefe_sec_2010', 'h_jefe_sec_2010', 'hog_2010'),
    ('pct_jefe_uni_2010', 'h_jefe_uni_2010', 'hog_2010'),
    ('pct_domestic_2010', 'h_domestic_2010', 'hog_2010'),
    ('pct_hacinam_2010', 'h_hacinam_2010', 'hog_2010'),
    ('pct_cadena_2010', 'h_cadena_2010', 'hog_2010'),
]:
    c10_agg[var] = 100.0 * c10_agg[num] / c10_agg[den].replace(0, np.nan)
# Keep only percentages + key counts
c10_final = c10_agg[['dpto5', 'pob_2010', 'hog_2010', 'area_km2_2010'] +
    [c for c in c10_agg.columns if c.startswith('pct_')]].copy()
print(f"  Census 2010: {len(c10_final)} departments")

# ---
# Census 2010 — population (education, age)
# ---
print("Loading Census 2010 population...", flush=True)
p10 = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        SUM(p_total) as p_total_2010,
        SUM(p_18a29) as p_18a29_2010,
        SUM(p_30a54) as p_30a54_2010,
        SUM(p_18a) as p_18a_2010,
        SUM(p18a_unive) as p_univ_2010,
        SUM(p18a_secun) as p_sec_2010,
        SUM(p18a_prima) as p_prim_2010,
        SUM(p18a_sin_i) as p_sin_instr_2010
    FROM censo_2010.poblacion_2010
    GROUP BY SUBSTRING(redcode,1,5)
""")
p10_agg = p10.groupby('dpto5').agg(
    **{c: (c, 'sum') for c in p10.columns if c != 'dpto5'}
).reset_index()
p10_agg['pct_18a29_2010'] = 100.0 * p10_agg['p_18a29_2010'] / p10_agg['p_total_2010'].replace(0, np.nan)
p10_agg['pct_univ_pob_2010'] = 100.0 * p10_agg['p_univ_2010'] / p10_agg['p_18a_2010'].replace(0, np.nan)
p10_final = p10_agg[['dpto5', 'pct_18a29_2010', 'pct_univ_pob_2010']].copy()
print(f"  Census 2010 population: {len(p10_final)} departments")

# ---
# Census 2010 — estratificación (marginality)
# ---
print("Loading Census 2010 estratificación...", flush=True)
e10 = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        AVG(e) as entropy_2010,
        AVG(segmento) as segmento_2010,
        AVG(densidad) as densidad_2010,
        SUM(h_marginal) as h_marginal_2010,
        SUM(h_margin_1) as h_marginal2_2010,
        SUM(h_total) as h_total_e10
    FROM censo_2010.estratificacion_2010
    GROUP BY SUBSTRING(redcode,1,5)
""")
e10_agg = e10.groupby('dpto5').agg(
    entropy_2010=('entropy_2010', 'mean'),
    segmento_2010=('segmento_2010', 'mean'),
    densidad_2010=('densidad_2010', 'mean'),
    h_marginal_2010=('h_marginal_2010', 'sum'),
    h_marginal2_2010=('h_marginal2_2010', 'sum'),
    h_total_e10=('h_total_e10', 'sum'),
).reset_index()
e10_agg['pct_marginal_2010'] = 100.0 * e10_agg['h_marginal_2010'] / e10_agg['h_total_e10'].replace(0, np.nan)
e10_final = e10_agg[['dpto5', 'entropy_2010', 'segmento_2010', 'pct_marginal_2010']].copy()
print(f"  Estratificación 2010: {len(e10_final)} departments")

# ---
# Census 2022
# ---
print("Loading Census 2022...", flush=True)
c22 = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        MAX(prov) as prov_2022, MAX(dpto) as dpto_2022,
        SUM(radios_pob) as pob_2022,
        SUM(h_total) as hog_2022,
        SUM(h_nbi) as h_nbi_2022,
        SUM(h_computad) as h_pc_2022,
        SUM(h_celular) as h_cel_2022,
        SUM(h_cloaca) as h_cloaca_2022,
        SUM(h_combusti) as h_gas_2022,
        SUM(h_propieta) as h_prop_2022,
        SUM(h_inquilin) as h_inq_2022,
        SUM(h_jefe_muj) as h_jefa_muj_2022,
        SUM(h_jefe_sec) as h_jefe_sec_2022,
        SUM(h_jefe_uni) as h_jefe_uni_2022,
        SUM(h_domestic) as h_domestic_2022,
        SUM(h_hacinami) as h_hacinam_2022,
        SUM(h_cadena) as h_cadena_2022,
        SUM(radios_sup) as area_km2_2022
    FROM censo_2022.censo_nbi_2022
    GROUP BY SUBSTRING(redcode,1,5)
""")
c22_agg = c22.groupby('dpto5').agg(
    prov_2022=('prov_2022', 'first'),
    dpto_2022=('dpto_2022', 'first'),
    **{c: (c, 'sum') for c in c22.columns if c not in ('dpto5','prov_2022','dpto_2022')}
).reset_index()
for var, num, den in [
    ('pct_nbi_2022', 'h_nbi_2022', 'hog_2022'),
    ('pct_pc_2022', 'h_pc_2022', 'hog_2022'),
    ('pct_cel_2022', 'h_cel_2022', 'hog_2022'),
    ('pct_cloaca_2022', 'h_cloaca_2022', 'hog_2022'),
    ('pct_gas_2022', 'h_gas_2022', 'hog_2022'),
    ('pct_prop_2022', 'h_prop_2022', 'hog_2022'),
    ('pct_jefa_muj_2022', 'h_jefa_muj_2022', 'hog_2022'),
    ('pct_jefe_sec_2022', 'h_jefe_sec_2022', 'hog_2022'),
    ('pct_jefe_uni_2022', 'h_jefe_uni_2022', 'hog_2022'),
    ('pct_domestic_2022', 'h_domestic_2022', 'hog_2022'),
    ('pct_hacinam_2022', 'h_hacinam_2022', 'hog_2022'),
    ('pct_cadena_2022', 'h_cadena_2022', 'hog_2022'),
]:
    c22_agg[var] = 100.0 * c22_agg[num] / c22_agg[den].replace(0, np.nan)
c22_final = c22_agg[['dpto5', 'prov_2022', 'dpto_2022', 'pob_2022', 'hog_2022', 'area_km2_2022'] +
    [c for c in c22_agg.columns if c.startswith('pct_')]].copy()
print(f"  Census 2022: {len(c22_final)} departments")

# ---
# Census 2022 — population (education, age)
# ---
print("Loading Census 2022 population...", flush=True)
p22 = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        SUM(p_total) as p_total_2022,
        SUM(p18a_unive) as p_univ_2022,
        SUM(p18a_secun) as p_sec_2022,
        SUM(p_total) FILTER (WHERE TRUE) as p_all
    FROM censo_2022.censo_poblacion_2022
    GROUP BY SUBSTRING(redcode,1,5)
""")
p22_agg = p22.groupby('dpto5').agg(
    **{c: (c, 'sum') for c in p22.columns if c != 'dpto5'}
).reset_index()
p22_final = p22_agg[['dpto5', 'p_total_2022']].copy()
print(f"  Census 2022 population: {len(p22_final)} departments")

# ---
# Census 2022 — employment
# ---
print("Loading Census 2022 employment...", flush=True)
try:
    emp22 = read_and_map("""
        SELECT SUBSTRING(redcode,1,5) as dpto5,
            SUM(p_total) as p_total_emp,
            SUM(COALESCE(ocupados,0)) as ocupados_2022,
            SUM(COALESCE(desocupados,0)) as desocupados_2022,
            SUM(COALESCE(activos,0)) as activos_2022,
            SUM(COALESCE(inactivos,0)) as inactivos_2022
        FROM censo_2022.censo_empleo_2022
        GROUP BY SUBSTRING(redcode,1,5)
    """)
    emp22_agg = emp22.groupby('dpto5').agg(
        **{c: (c, 'sum') for c in emp22.columns if c != 'dpto5'}
    ).reset_index()
    emp22_agg['tasa_actividad_2022'] = 100.0 * emp22_agg['activos_2022'] / emp22_agg['p_total_emp'].replace(0, np.nan)
    emp22_agg['tasa_desocup_2022'] = 100.0 * emp22_agg['desocupados_2022'] / emp22_agg['activos_2022'].replace(0, np.nan)
    emp22_final = emp22_agg[['dpto5', 'tasa_actividad_2022', 'tasa_desocup_2022']].copy()
    print(f"  Employment 2022: {len(emp22_final)} departments")
except Exception as e:
    print(f"  Employment 2022: skipped ({e})")
    emp22_final = pd.DataFrame(columns=['dpto5', 'tasa_actividad_2022', 'tasa_desocup_2022'])

# ---
# Nighttime lights (annual means per department)
# ---
print("Loading NTL...", flush=True)
ntl = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        COUNT(*) as n_radios_ntl,
        AVG((rad_2014_03+rad_2014_06+rad_2014_09+rad_2014_12)/4.0) as rad_2014,
        AVG((rad_2015_03+rad_2015_06+rad_2015_09+rad_2015_12)/4.0) as rad_2015,
        AVG((rad_2016_03+rad_2016_06+rad_2016_09+rad_2016_12)/4.0) as rad_2016,
        AVG((rad_2017_03+rad_2017_06+rad_2017_09+rad_2017_12)/4.0) as rad_2017,
        AVG((rad_2018_03+rad_2018_06+rad_2018_09+rad_2018_12)/4.0) as rad_2018,
        AVG((rad_2019_03+rad_2019_06+rad_2019_09+rad_2019_12)/4.0) as rad_2019,
        AVG((rad_2020_03+rad_2020_06+rad_2020_09+rad_2020_12)/4.0) as rad_2020,
        AVG((rad_2021_03+rad_2021_06+rad_2021_09+rad_2021_12)/4.0) as rad_2021,
        AVG((rad_2022_03+rad_2022_06+rad_2022_09+rad_2022_12)/4.0) as rad_2022,
        AVG((rad_2023_03+rad_2023_06+rad_2023_09+rad_2023_12)/4.0) as rad_2023,
        AVG((rad_2024_03+rad_2024_06+rad_2024_09+rad_2024_12)/4.0) as rad_2024,
        AVG((rad_2025_03+rad_2025_06+rad_2025_09+rad_2025_12)/4.0) as rad_2025,
        STDDEV((rad_2022_03+rad_2022_06+rad_2022_09+rad_2022_12)/4.0) /
            NULLIF(AVG((rad_2022_03+rad_2022_06+rad_2022_09+rad_2022_12)/4.0), 0) as rad_cv_2022
    FROM gee_2022.luces_promedio_estacional
    GROUP BY SUBSTRING(redcode,1,5)
""")
# For mapped departments (CABA etc), we need weighted average by n_radios
ntl_weighted = []
for dpto5, grp in ntl.groupby('dpto5'):
    row = {'dpto5': dpto5, 'n_radios_ntl': grp['n_radios_ntl'].sum()}
    weights = grp['n_radios_ntl']
    for col in [c for c in grp.columns if c.startswith('rad_')]:
        row[col] = np.average(grp[col].values, weights=weights)
    ntl_weighted.append(row)
ntl_final = pd.DataFrame(ntl_weighted)
ntl_final['rad_delta_2014_2024'] = ntl_final['rad_2024'] - ntl_final['rad_2014']
ntl_final['rad_mean_2022'] = ntl_final['rad_2022']
print(f"  NTL: {len(ntl_final)} departments")

# ---
# NDVI
# ---
print("Loading NDVI...", flush=True)
ndvi = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        AVG(ndvi_p90_2022) as ndvi_p90_2022,
        AVG(ndvi_p10_2022) as ndvi_p10_2022
    FROM gee_2022.ndvi_argentina_deciles
    GROUP BY SUBSTRING(redcode,1,5)
""")
ndvi_agg = ndvi.groupby('dpto5').agg(
    ndvi_p90_2022=('ndvi_p90_2022', 'mean'),
    ndvi_p10_2022=('ndvi_p10_2022', 'mean'),
).reset_index()
ndvi_agg['ndvi_range_2022'] = ndvi_agg['ndvi_p90_2022'] - ndvi_agg['ndvi_p10_2022']
print(f"  NDVI: {len(ndvi_agg)} departments")

# ---
# GitHub
# ---
print("Loading GitHub...", flush=True)
gh = pd.read_sql("""
    SELECT LPAD(redcode::text,5,'0') as dpto5,
        provincia, departamento,
        total_developers, total_repos, total_original_repos,
        total_forks, avg_repos_per_dev, total_stars, total_forks_received,
        pct_hireable, pct_web_development, pct_mobile, pct_data_science,
        pct_systems, pct_enterprise,
        language_diversity_index, pielou_evenness, hhi_languages,
        simpson_diversity, hill_q0_richness, hill_q1_shannon, hill_q2_simpson,
        cr4_languages, theil_languages,
        gini_repos_per_dev, cv_repos_per_dev, gini_stars_per_dev,
        hhi_developers, cr4_developers,
        pct_archived, pct_with_license, pct_with_stars, total_size_mb,
        repos_2008, repos_2009, repos_2010, repos_2011, repos_2012, repos_2013,
        repos_2014, repos_2015, repos_2016, repos_2017, repos_2018, repos_2019,
        repos_2020, repos_2021, repos_2022, repos_2023, repos_2024, repos_2025, repos_2026
    FROM github_argentina.departamentos_geo
""", engine)
gh['dpto5'] = gh['dpto5'].apply(map_dpto)
gh = gh[gh['dpto5'].notna()].copy()

# For CABA: aggregate. For others: pass through.
# Numeric cols to sum
sum_cols = ['total_developers', 'total_repos', 'total_original_repos', 'total_forks',
            'total_stars', 'total_forks_received', 'total_size_mb'] + \
           [f'repos_{y}' for y in range(2008, 2027)]
# Weighted average cols (weight by total_developers)
wavg_cols = ['pct_hireable', 'pct_web_development', 'pct_mobile', 'pct_data_science',
             'pct_systems', 'pct_enterprise', 'language_diversity_index', 'pielou_evenness',
             'hhi_languages', 'simpson_diversity', 'hill_q0_richness', 'hill_q1_shannon',
             'hill_q2_simpson', 'cr4_languages', 'theil_languages', 'gini_repos_per_dev',
             'cv_repos_per_dev', 'gini_stars_per_dev', 'hhi_developers', 'cr4_developers',
             'pct_archived', 'pct_with_license', 'pct_with_stars', 'avg_repos_per_dev']

gh_rows = []
for dpto5, grp in gh.groupby('dpto5'):
    row = {'dpto5': dpto5, 'gh_provincia': grp['provincia'].iloc[0], 'gh_departamento': grp['departamento'].iloc[0]}
    for c in sum_cols:
        row[c] = grp[c].sum()
    weights = grp['total_developers'].replace(0, np.nan)
    for c in wavg_cols:
        valid = grp[c].notna() & weights.notna()
        if valid.any() and weights[valid].sum() > 0:
            row[c] = np.average(grp.loc[valid, c], weights=weights[valid])
        else:
            row[c] = np.nan
    # Recalculate avg_repos_per_dev
    if row['total_developers'] > 0:
        row['avg_repos_per_dev'] = row['total_repos'] / row['total_developers']
    gh_rows.append(row)
gh_final = pd.DataFrame(gh_rows)
# Prefix GitHub columns
rename = {c: f'gh_{c}' for c in gh_final.columns if c not in ('dpto5', 'gh_provincia', 'gh_departamento')}
gh_final = gh_final.rename(columns=rename)
print(f"  GitHub: {len(gh_final)} departments, {gh_final['gh_total_developers'].sum():.0f} total devs")

# ---
# ENACOM
# ---
print("Loading ENACOM...", flush=True)
enacom_tech = read_and_map("""
    SELECT redcode as dpto5,
        SUM(accesos) as inet_total,
        SUM(CASE WHEN tecnologia='FIBRA OPTICA' THEN accesos ELSE 0 END) as inet_fibra,
        SUM(CASE WHEN tecnologia='CABLEMODEM' THEN accesos ELSE 0 END) as inet_cable,
        SUM(CASE WHEN tecnologia='WIRELESS' THEN accesos ELSE 0 END) as inet_wireless,
        SUM(CASE WHEN tecnologia='ADSL' THEN accesos ELSE 0 END) as inet_adsl,
        COUNT(DISTINCT tecnologia) as enacom_n_tech
    FROM enacom.internet_tecnologia_localidades
    WHERE redcode != '00000'
    GROUP BY redcode
""")
enacom_agg = enacom_tech.groupby('dpto5').agg(
    inet_total=('inet_total', 'sum'),
    inet_fibra=('inet_fibra', 'sum'),
    inet_cable=('inet_cable', 'sum'),
    inet_wireless=('inet_wireless', 'sum'),
    inet_adsl=('inet_adsl', 'sum'),
    enacom_n_tech=('enacom_n_tech', 'max'),
).reset_index()
enacom_agg['pct_fibra'] = 100.0 * enacom_agg['inet_fibra'] / enacom_agg['inet_total'].replace(0, np.nan)

enacom_vel = read_and_map("""
    SELECT redcode as dpto5,
        SUM(CASE WHEN velocidad_mbps >= 100 THEN accesos ELSE 0 END)::float /
            NULLIF(SUM(accesos),0) * 100 as pct_fast_internet,
        SUM(CASE WHEN velocidad_mbps < 10 THEN accesos ELSE 0 END)::float /
            NULLIF(SUM(accesos),0) * 100 as pct_slow_internet
    FROM enacom.internet_velocidad_localidades
    WHERE redcode != '00000'
    GROUP BY redcode
""")
enacom_vel_agg = enacom_vel.groupby('dpto5').agg(
    pct_fast_internet=('pct_fast_internet', 'mean'),
    pct_slow_internet=('pct_slow_internet', 'mean'),
).reset_index()
enacom_final = enacom_agg.merge(enacom_vel_agg, on='dpto5', how='left')
print(f"  ENACOM: {len(enacom_final)} departments")

# ---
# Barrios populares
# ---
print("Loading Barrios Populares...", flush=True)
bp = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        COUNT(*) as n_barrios_populares,
        SUM(familias) as familias_bp
    FROM otros_2022."Registro Nacional de Barrios Populares 2023"
    GROUP BY SUBSTRING(redcode,1,5)
""")
bp_agg = bp.groupby('dpto5').agg(
    n_barrios_populares=('n_barrios_populares', 'sum'),
    familias_bp=('familias_bp', 'sum'),
).reset_index()
print(f"  Barrios Populares: {len(bp_agg)} departments")

# ---
# Exclusion index 2010
# ---
print("Loading exclusion 2010...", flush=True)
exc10 = read_and_map("""
    SELECT SUBSTRING(redcode,1,5) as dpto5,
        AVG(incidencia) as excl_incidencia_2010,
        AVG(concentrac) as excl_concentrac_2010
    FROM censo_2010.exclusion_2010
    GROUP BY SUBSTRING(redcode,1,5)
""")
exc10_agg = exc10.groupby('dpto5').agg(
    excl_incidencia_2010=('excl_incidencia_2010', 'mean'),
    excl_concentrac_2010=('excl_concentrac_2010', 'mean'),
).reset_index()
print(f"  Exclusion 2010: {len(exc10_agg)} departments")

# ---
# Merge all into master table
# ---
print("\nMerging all sources...", flush=True)

# Start from Census 2022 (most complete)
master = c22_final.copy()

# Merge everything
for df, name in [
    (c10_final, 'Census 2010'),
    (p10_final, 'Population 2010'),
    (e10_final, 'Estratificación 2010'),
    (exc10_agg, 'Exclusion 2010'),
    (p22_final, 'Population 2022'),
    (emp22_final, 'Employment 2022'),
    (ntl_final, 'NTL'),
    (ndvi_agg, 'NDVI'),
    (gh_final, 'GitHub'),
    (enacom_final, 'ENACOM'),
    (bp_agg, 'Barrios Populares'),
]:
    if not df.empty:
        master = master.merge(df, on='dpto5', how='left')
        print(f"  + {name}: {len(df)} rows merged")

# Derived variables
master['region'] = master['dpto5'].apply(get_region)
master['densidad_2022'] = master['pob_2022'] / master['area_km2_2022'].replace(0, np.nan)
master['gh_devs_per_10k'] = master['gh_total_developers'].fillna(0) * 10000 / master['pob_2022'].replace(0, np.nan)
master['gh_has_devs'] = (master['gh_total_developers'].fillna(0) > 0).astype(int)
master['inet_penetracion_hog'] = master['inet_total'].fillna(0) * 100 / master['hog_2022'].replace(0, np.nan)
master['pct_familias_bp'] = master['familias_bp'].fillna(0) * 100 / master['hog_2022'].replace(0, np.nan)

# Change variables 2010 → 2022
master['delta_nbi'] = master['pct_nbi_2022'] - master['pct_nbi_2010']
master['delta_pc'] = master['pct_pc_2022'] - master['pct_pc_2010']
master['delta_jefe_uni'] = master['pct_jefe_uni_2022'] - master['pct_jefe_uni_2010']
master['delta_cloaca'] = master['pct_cloaca_2022'] - master['pct_cloaca_2010']
master['pop_growth_pct'] = (master['pob_2022'] / master['pob_2010'].replace(0, np.nan) - 1) * 100

# Fill GitHub NAs with 0 for departments with no developers
gh_fill_cols = [c for c in master.columns if c.startswith('gh_') and c not in ('gh_provincia', 'gh_departamento')]
for c in gh_fill_cols:
    if c in ('gh_total_developers', 'gh_total_repos', 'gh_total_stars', 'gh_has_devs',
             'gh_devs_per_10k', 'gh_total_original_repos', 'gh_total_forks',
             'gh_total_forks_received', 'gh_total_size_mb'):
        master[c] = master[c].fillna(0)
    # Leave diversity/inequality indices as NaN for 0-developer departments

# Rename identifier columns
master = master.rename(columns={'prov_2022': 'provincia', 'dpto_2022': 'departamento'})

# Use provincia from GitHub for CABA (Census says "Ciudad Autónoma de Buenos Aires")
caba_mask = master['dpto5'] == '02000'
if caba_mask.any():
    master.loc[caba_mask, 'departamento'] = 'CABA'

print(f"\n  MASTER TABLE: {len(master)} departments, {len(master.columns)} columns")
print(f"  With GitHub developers: {(master['gh_has_devs']==1).sum()}")
print(f"  Without developers: {(master['gh_has_devs']==0).sum()}")

# ---
# Sequence table (dpto x year with NTL + GitHub repos)
# ---
print("\nBuilding sequence table...", flush=True)
years = list(range(2014, 2026))
seq_rows = []
for _, row in master.iterrows():
    for yr in years:
        seq_rows.append({
            'dpto5': row['dpto5'],
            'year': yr,
            'rad': row.get(f'rad_{yr}', np.nan),
            'repos': row.get(f'gh_repos_{yr}', 0) if pd.notna(row.get(f'gh_repos_{yr}')) else 0,
        })
seq = pd.DataFrame(seq_rows)

# Compute annual terciles for radiance
for yr in years:
    mask = seq['year'] == yr
    vals = seq.loc[mask, 'rad'].dropna()
    t1 = vals.quantile(0.333)
    t2 = vals.quantile(0.667)
    seq.loc[mask, 'rad_tercile'] = pd.cut(
        seq.loc[mask, 'rad'], bins=[-np.inf, t1, t2, np.inf],
        labels=['low', 'mid', 'high']
    )

# GitHub: active/inactive
seq['gh_active'] = (seq['repos'] > 0).astype(int)

# Combined state (6 states)
def make_state(row):
    rad = row['rad_tercile']
    gh = 'Active' if row['gh_active'] == 1 else 'Inactive'
    if pd.isna(rad):
        return np.nan
    prefix = {'low': 'Dark', 'mid': 'Dim', 'high': 'Bright'}[rad]
    return f'{prefix}-{gh}'

seq['state'] = seq.apply(make_state, axis=1)
print(f"  Sequence table: {len(seq)} rows ({len(master)} depts × {len(years)} years)")
print(f"  State distribution:")
print(seq['state'].value_counts().to_string())

# ---
# Write to PostgreSQL
# ---
print("\nWriting to PostgreSQL...", flush=True)

# Drop geometry columns if present (we'll add geometry via join later)
for col in ['geometry', 'geom']:
    if col in master.columns:
        master = master.drop(columns=[col])

master.to_sql('departamentos', engine, schema='art1', if_exists='replace', index=False)
print(f"  art1.departamentos: {len(master)} rows, {len(master.columns)} columns")

seq.to_sql('secuencias', engine, schema='art1', if_exists='replace', index=False)
print(f"  art1.secuencias: {len(seq)} rows")

# Create a convenience view with geometry
with engine.connect() as conn:
    conn.execute(text("""
        CREATE OR REPLACE VIEW art1.departamentos_geo AS
        SELECT d.*, g.geometry
        FROM art1.departamentos d
        LEFT JOIN public.departamentos_argentina g ON d.dpto5 = g.redcode
    """))
    conn.commit()
print("  art1.departamentos_geo: view created")

print("\n=== DONE ===")
print(f"Schema art1 ready with:")
print(f"  - departamentos: {len(master)} rows × {len(master.columns)} cols")
print(f"  - secuencias: {len(seq)} rows (for Sequence Analysis)")
print(f"  - departamentos_geo: view with geometry")
