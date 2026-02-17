# The Spatiality of Software: Subnational Economic Complexity from GitHub Data in Argentina

**Author:** Raimundo Elias Gomez
**Affiliations:** CONICET / National University of Misiones (Argentina); Faculty of Arts, University of Porto (Portugal)
**Contact:** elias.gomez@conicet.gov.ar
**ORCID:** [0000-0002-4468-9618](https://orcid.org/0000-0002-4468-9618)

---

## Overview

This repository contains the data, analysis scripts, and figures for the article *"The spatiality of software: subnational economic complexity from GitHub data in Argentina"*, currently under peer review.

The study constructs an Economic Complexity Index for software production (ECI<sub>software</sub>) at the level of 224 Argentine departments using a bipartite network of departments and 87 programming languages derived from 229,270 geocoded GitHub repositories. A three-stage analytical strategy — Multiple Correspondence Analysis (MCA), Hierarchical Agglomerative Clustering (CAH), and type-specific regressions — examines how the determinants of software complexity vary across six territorial types.

## Repository structure

```
github-subir/
├── README.md
├── data/                         # Processed datasets and summary tables
│   ├── departments_full.csv      # All 511 departments: MCA coords, clusters, ECI, census vars
│   ├── bipartite_matrix.csv      # 224 depts x 87 languages (repo counts, filtered)
│   ├── rca_binary_matrix.csv     # 224 x 87 binary RCA matrix (threshold >= 1)
│   ├── eci_ranking_FINAL.csv     # ECI ranking for 224 departments
│   ├── table_01_eci_ranking_full.csv     # ECI ranking with sociodemographic variables
│   ├── table_02_pci_ranking_languages.csv # PCI ranking for 87 programming languages
│   ├── table_03_cluster_profiles.csv      # Mean profiles of 6 departmental types
│   ├── table_04_regression_summary.csv    # Regression coefficients by type
│   ├── table_05_key_numbers.csv           # Summary statistics (key-value)
│   ├── table_06_crossvalidation_geo.csv   # Geospatial cross-validation (511 depts)
│   └── regression_output_FINAL.txt        # Full regression output (text)
├── figures/                      # Article figures (300 DPI)
│   ├── fig_01_pci_ubiquity.png           # Figure 1: PCI vs ubiquity (87 languages)
│   ├── fig_02_mca_biplot.png             # Figure 2: MCA biplot (Axes 1-2, N=511)
│   ├── fig_03_cah_mca_clusters.png       # Figure 3: Six types in MCA space
│   ├── fig_04_cluster_maps.png           # Figure 4: Spatial distribution of types
│   ├── fig_05_eci_vs_devs.png            # Figure 5: ECI vs developer density
│   ├── fig_06_forest_plot.png            # Figure 6: Forest plot of betas by type
│   ├── fig_S1_dendrogram.png             # Figure S1: Ward's dendrogram (k=6)
│   └── fig_S2_diagnostics_panel.png      # Figure S2: MCA scree + cluster quality
├── scripts/                      # Analysis pipeline (Python)
│   ├── 00_build_schema.py        # Stage 0: Integrate 11 data sources into art1 schema
│   ├── 01_compute_eci.py         # Stage 1: Compute ECI via eigenvalue decomposition
│   ├── 02_mca.py                 # Stage 2a: Multiple Correspondence Analysis (8 vars, N=511)
│   ├── 03_cah.py                 # Stage 2b: Ward's CAH on MCA coordinates (k=6)
│   ├── 04_regressions_by_type.py # Stage 3: Pooled + type-specific regressions, Chow test
│   ├── 05_regenerate_figures.py  # Generate all 8 figures (6 article + 2 supplementary)
│   └── 06_cluster_maps.py       # Generate Figure 4 (3x2 small-multiples map)
├── audit/                        # Data quality and geocoding validation
│   ├── audit_01_full_province_department.csv  # Raw vs geo-validated counts (513 depts)
│   ├── audit_02_discrepancies.csv             # 32 departments with discrepancies
│   ├── audit_03_province_summary.csv          # Province-level data integrity summary
│   ├── audit_04_foreign_users.csv             # 76 excluded non-Argentine users
│   ├── audit_05_foreign_repos_by_dept.csv     # Departments affected by foreign repos
│   ├── audit_06_ambiguous_users_sample.csv    # 31 ambiguous location samples
│   └── audit_07_eci_before_after.csv          # ECI ranking before/after corrections
└── supplementary/                # Supplementary material
    ├── supplementary_tables.md              # Supplementary tables and figures
    ├── table_S1_eci_full_ranking.csv        # Full ECI ranking (224 departments)
    ├── table_S2_cluster_region_crosstab.csv # Cluster × region cross-tabulation
    ├── table_S3_small_types_data.csv        # Data for small-N types (Peripheral, Semi-Rural)
    └── table_S4_within_type_correlations.csv # Within-type correlations with ECI
```

## Data description

### Core datasets

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `departments_full.csv` | 511 | 28 | All Argentine departments with census (2010), MCA coordinates (5 dims), cluster assignment, ECI, GitHub metrics |
| `bipartite_matrix.csv` | 224 | 88 | Repository counts by department and programming language (dpto5 + 87 languages) |
| `rca_binary_matrix.csv` | 224 | 88 | Binarised Revealed Comparative Advantage (RCA >= 1) |
| `table_02_pci_ranking_languages.csv` | 87 | 5 | Product Complexity Index for programming languages |

### Key variables in `departments_full.csv`

| Variable | Source | Description |
|----------|--------|-------------|
| `dpto5` | INDEC | Five-digit department code |
| `region` | Derived | Six regions: CABA, Pampeana, NOA, NEA, Cuyo, Patagonia |
| `pob_2010`, `pob_2022` | Census | Population |
| `pct_jefe_sec_2010` | Census 2010 | % household heads with secondary education |
| `pct_jefe_uni_2010` | Census 2010 | % household heads with university education |
| `pct_pc_2010` | Census 2010 | % households with computer |
| `pct_nbi_2010` | Census 2010 | % with unsatisfied basic needs (poverty) |
| `pct_hacinam_2010` | Census 2010 | % overcrowding |
| `rad_2014` | VIIRS | Mean nighttime radiance (2014) |
| `tasa_empleo_2010` | Census 2010 | Employment rate |
| `mca_dim1`...`mca_dim5` | MCA | Factorial coordinates (5 retained axes) |
| `mca_cluster` | CAH | Cluster number (1-6) |
| `mca_cluster_label` | CAH | Cluster label |
| `eci_software` | ECI | Economic Complexity Index (standardised) |
| `eci_diversity` | ECI | Number of languages with RCA >= 1 |
| `eci_avg_ubiquity` | ECI | Mean ubiquity of RCA languages |
| `gh_total_developers` | GitHub | Total geocoded developers |
| `gh_total_repos` | GitHub | Total repositories |
| `gh_devs_per_10k` | Derived | Developers per 10,000 inhabitants |
| `gh_hill_q1_shannon` | GitHub | Language diversity (Shannon entropy) |

## Analytical pipeline

The scripts are numbered in execution order and depend on a PostgreSQL database (`posadas`) with the source data. The pipeline proceeds as follows:

1. **`00_build_schema.py`** — Integrates 11 data sources (Census 2010/2022, VIIRS nighttime lights, NDVI, GitHub, ENACOM) into a single analysis-ready table (`art1.departamentos`, 511 departments, ~208 columns).

2. **`01_compute_eci.py`** — Constructs the bipartite network (departments x languages), computes RCA, and extracts ECI and PCI via eigenvalue decomposition of the normalised adjacency matrix. Applies geocoding corrections (Cordoba shift, CABA aggregation, foreign user exclusion).

3. **`02_mca.py`** — Multiple Correspondence Analysis on 8 pre-treatment variables discretised into terciles (24 modalities, N=511). Retains 5 axes via Benzecri correction. Projects ECI and developer metrics as supplementary variables.

4. **`03_cah.py`** — Ward's hierarchical clustering on 5 MCA coordinates. Selects k=6 (silhouette=0.330, Calinski-Harabasz=224.5). Profiles clusters with ANOVA and chi-squared tests.

5. **`04_regressions_by_type.py`** — Pooled and type-specific OLS regressions of ECI on pre-treatment predictors. Chow test for structural heterogeneity. Forest plot of standardised coefficients.

6. **`05_regenerate_figures.py`** — Generates all 8 figures (6 article + 2 supplementary) with unified formatting (300 DPI).

7. **`06_cluster_maps.py`** — Generates Figure 4 (3x2 small-multiples map of cluster spatial distribution) using PostGIS geometries.

## Key findings

- **ECI<sub>software</sub> is distinct from developer counts**: r = 0.47 (moderate correlation)
- **PCI validates the framework**: scientific computing languages (Erlang, Fortran, Julia) rank as most complex; web technologies (JavaScript, HTML, CSS) as least complex
- **Six departmental types** explain 30.2% of ECI variance (eta-squared = 0.302)
- **Determinants are structurally heterogeneous**: education drives complexity in Metropolitan-Core; computer ownership in Metropolitan-Diversified; population alone in Pampeana-Educated; no predictor significant in Intermediate-Urban

## Data sources

| Source | Period | Coverage | Access |
|--------|--------|----------|--------|
| GitHub API | Accumulated through 2025 | 229,270 repos, 23,619 users | Scraped early 2026 |
| Census (INDEC) | 2010, 2022 | 511 departments | [datos.gob.ar](https://datos.gob.ar) |
| VIIRS DNB | 2014 | Department-level radiance | [Google Earth Engine](https://earthengine.google.com) |
| ENACOM | ~2023 | Internet infrastructure | [datosabiertos.enacom.gob.ar](https://datosabiertos.enacom.gob.ar) |

## Requirements

```
python >= 3.10
numpy
pandas
scipy
scikit-learn
prince
matplotlib
seaborn
geopandas
sqlalchemy
psycopg2
```

## Citation

If you use these data or methods, please cite:

> Gomez, R. E. (2026). The spatiality of software: subnational economic complexity from GitHub data in Argentina. *Working paper*.

**Zenodo DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18674718.svg)](https://doi.org/10.5281/zenodo.18674718)

## Licence

Data and code are provided under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
