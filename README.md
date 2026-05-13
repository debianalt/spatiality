# Software complexity and the accumulation of technical capabilities within a peripheral economy: a subnational typology of Argentina

**Author:** Raimundo Elias Gomez
**Affiliation:** CONICET / Facultad de Humanidades y Ciencias Sociales, Universidad Nacional de Misiones (UNaM)
**Contact:** lsgomez001@gmail.com
**ORCID:** [0000-0002-4468-9618](https://orcid.org/0000-0002-4468-9618)

---

## Overview

This repository contains the data, analysis scripts, and figures for the article *"Software complexity and the accumulation of technical capabilities within a peripheral economy: a subnational typology of Argentina"*, submitted to *Regional Studies* in May 2026 (previously submitted to the *Journal of Economic Geography*, February 2026, JOEG-2026-083).

The study constructs an Economic Complexity Index for software production (ECI<sub>software</sub>) at the level of 224 Argentine departments using a bipartite network of departments and 87 programming languages derived from 229,270 geocoded GitHub repositories. A three-stage analytical strategy — Multiple Correspondence Analysis (MCA), Hierarchical Agglomerative Clustering (CAH), and type-specific regressions — examines how the determinants of software complexity vary across six territorial types. Two robustness checks are included: (1) a bundle-based ECI recomputed after mapping individual languages onto the 38 software clusters identified by Juhász et al. (2026), confirming Pearson *r* = 0.90 with the main measure; and (2) a longitudinal stability analysis computing ECI for cumulative annual snapshots from 2015 to 2025, showing consecutive-year rank correlations of 0.82–0.97 and 68 per cent top-quintile persistence from 2020 to 2025.

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
│   ├── regression_output_FINAL.txt        # Full regression output (text)
│   ├── juhasz_language_clusters.csv       # Juhász et al. (2026) language-to-bundle mapping
│   ├── table_s_bundle_robustness.csv      # Table S5: ECI_individual vs ECI_bundle (224 depts)
│   └── eci_panel_2015_2025.csv            # Longitudinal ECI panel (1,504 dept-year obs)
├── figures/                      # Article figures (300 DPI)
│   ├── fig_01_pci_ubiquity.png           # Figure 1: PCI vs ubiquity (87 languages)
│   ├── fig_02_mca_biplot.png             # Figure 2: MCA biplot (Axes 1-2, N=511)
│   ├── fig_03_cah_mca_clusters.png       # Figure 3: Six types in MCA space
│   ├── fig_04_cluster_maps.png           # Figure 4: Spatial distribution of types
│   ├── fig_05_eci_vs_devs.png            # Figure 5: ECI vs developer density
│   ├── fig_06_forest_plot.png            # Figure 6: Forest plot of betas by type
│   ├── fig_S1_dendrogram.png             # Figure S1: Ward's dendrogram (k=6)
│   ├── fig_S2_diagnostics_panel.png      # Figure S2: MCA scree + cluster quality
│   ├── fig_panel_trajectories.png        # Figure S3: ECI trajectories 2015-2025 by type
│   └── fig_panel_rank_stability.png      # Figure S3 supplement: Spearman rho heatmap
├── scripts/                      # Analysis pipeline (Python)
│   ├── 00_build_schema.py        # Stage 0: Integrate data sources into art1 schema
│   ├── 01_compute_eci.py         # Stage 1: Compute ECI via eigenvalue decomposition
│   ├── 02_mca.py                 # Stage 2a: Multiple Correspondence Analysis (8 vars, N=511)
│   ├── 03_cah.py                 # Stage 2b: Ward's CAH on MCA coordinates (k=6)
│   ├── 04_regressions_by_type.py # Stage 3: Pooled + type-specific regressions, Chow test
│   ├── 05_regenerate_figures.py  # Generate all 8 figures (6 article + 2 supplementary)
│   ├── 06_cluster_maps.py        # Generate Figure 4 (3x2 small-multiples map)
│   ├── 07_correlation_table.py   # Generate Table S6: predictor correlation matrix
│   ├── 08_compute_eci_bundles.py # Robustness: bundle-based ECI (Juhász et al. 2026)
│   └── 09_eci_temporal_panel.py  # Robustness: longitudinal ECI panel 2015-2025
├── audit/                        # Data quality and geocoding validation
│   ├── audit_01_full_province_department.csv
│   ├── audit_02_discrepancies.csv
│   ├── audit_03_province_summary.csv
│   ├── audit_04_foreign_users.csv         # 76 excluded non-Argentine users
│   ├── audit_05_foreign_repos_by_dept.csv
│   ├── audit_06_ambiguous_users_sample.csv
│   └── audit_07_eci_before_after.csv      # ECI ranking before/after corrections
└── supplementary/                # Supplementary material
    ├── supplementary_tables.md              # Supplementary tables and figures (narrative)
    ├── table_S1_eci_full_ranking.csv        # Full ECI ranking (224 departments)
    ├── table_S2_cluster_region_crosstab.csv # Cluster × region cross-tabulation
    ├── table_S3_small_types_data.csv        # Data for small-N types (Peripheral, Semi-Rural)
    ├── table_S4_within_type_correlations.csv # Within-type correlations with ECI
    ├── table_s_bundle_robustness.csv        # Table S5: bundle robustness check
    └── table_S5_correlation_matrix.csv      # Table S6: predictor correlation matrix
```

## Data description

### Core datasets

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `departments_full.csv` | 511 | 28 | All Argentine departments with census (2010), MCA coordinates (5 dims), cluster assignment, ECI, GitHub metrics |
| `bipartite_matrix.csv` | 224 | 88 | Repository counts by department and programming language (dpto5 + 87 languages) |
| `rca_binary_matrix.csv` | 224 | 88 | Binarised Revealed Comparative Advantage (RCA >= 1) |
| `eci_panel_2015_2025.csv` | 1,504 | 7 | Longitudinal ECI panel: dept × year (2015–2025), cumulative snapshots |
| `table_s_bundle_robustness.csv` | 224 | 8 | Individual vs bundle ECI comparison |

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

## Analytical pipeline

The scripts are numbered in execution order and depend on a PostgreSQL database (`posadas`) with the source data.

1. **`00_build_schema.py`** — Integrates data sources (Census 2010, VIIRS nighttime lights, GitHub) into a single analysis-ready table (`art1.departamentos`, 511 departments).

2. **`01_compute_eci.py`** — Constructs the bipartite network (departments × languages), computes RCA, and extracts ECI and PCI via eigenvalue decomposition. Applies geocoding corrections (Córdoba shift, CABA aggregation, foreign user exclusion).

3. **`02_mca.py`** — Multiple Correspondence Analysis on 8 pre-treatment variables discretised into terciles (24 modalities, N=511). Retains 5 axes via Benzécri correction. Projects ECI and developer metrics as supplementary variables.

4. **`03_cah.py`** — Ward's hierarchical clustering on 5 MCA coordinates. Selects k=6 (silhouette=0.330, Caliński-Harabasz=224.5). Profiles clusters with ANOVA and chi-squared tests.

5. **`04_regressions_by_type.py`** — Pooled and type-specific OLS regressions of ECI on pre-treatment predictors. Chow test for structural heterogeneity. Forest plot of standardised coefficients.

6. **`05_regenerate_figures.py`** — Generates all 8 figures (6 article + 2 supplementary) at 300 DPI.

7. **`06_cluster_maps.py`** — Generates Figure 4 (3×2 small-multiples map of cluster spatial distribution).

8. **`07_correlation_table.py`** — Generates Table S6: pairwise Pearson correlations among predictor variables.

9. **`08_compute_eci_bundles.py`** — Robustness check: downloads the Juhász et al. (2026) language-to-cluster mapping, maps 87 Argentine languages onto 38 bundle clusters, recomputes ECI, and reports Pearson *r* and Spearman rho between bundle-based and individual-language ECI. Exports `table_s_bundle_robustness.csv`.

10. **`09_eci_temporal_panel.py`** — Robustness check: computes ECI for cumulative annual snapshots 2015–2025 using `created_at` timestamps in the repository database. Reports consecutive-year and multi-year rank stability (Spearman rho), quintile persistence, and exports `eci_panel_2015_2025.csv`. Generates Figure S3.

## Key findings

- **ECI<sub>software</sub> is distinct from developer counts**: *r* = 0.47 (moderate correlation)
- **PCI validates the framework**: scientific computing languages (Erlang, Fortran, Julia) rank as most complex; web technologies (JavaScript, HTML, CSS) as least complex
- **Six departmental types** explain 30.2% of ECI variance (eta-squared = 0.302, ANOVA *F* = 18.84, *p* < 0.001)
- **Determinants are structurally heterogeneous**: education drives complexity in Metropolitan-Core; computer ownership in Metropolitan-Diversified; population alone in Pampeana-Educated; no predictor significant in Intermediate-Urban
- **Bundle robustness**: ECI recomputed on 38 language bundles (Juhász et al., 2026) correlates at Pearson *r* = 0.90, Spearman rho = 0.90 with the individual-language measure
- **Longitudinal stability**: consecutive-year Spearman rank correlations 0.82–0.97 (2015–2025); ECI_2020 vs ECI_2025 rho = 0.781 (*p* < 0.001, N = 125); 68% of top-quintile departments in 2020 remain in the top quintile in 2025

## Data sources

| Source | Period | Coverage | Access |
|--------|--------|----------|--------|
| GitHub API | Accumulated through early 2026 | 229,270 repos, 23,619 users | Scraped January–February 2026 |
| Census (INDEC) | 2010 | 511 departments | [datos.gob.ar](https://datos.gob.ar) |
| VIIRS DNB | 2014 | Department-level radiance | [Google Earth Engine](https://earthengine.google.com) |
| Juhász et al. (2026) | — | 142 languages → 38 bundle clusters | [github.com/sandorjuhasz/eci_software](https://github.com/sandorjuhasz/eci_software) |

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
requests
```

## Citation

If you use these data or methods, please cite:

> Gomez, R. E. (2026). Software complexity and the accumulation of technical capabilities within a peripheral economy: a subnational typology of Argentina. *Submitted to Regional Studies*.

**Zenodo DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18674718.svg)](https://doi.org/10.5281/zenodo.18674718)

## Licence

Data and code are provided under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
