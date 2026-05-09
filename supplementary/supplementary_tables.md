# Supplementary Material

## The Spatiality of Software: Subnational Economic Complexity from GitHub Data in Argentina

**Author:** Raimundo Elias Gomez

---

### Figure S1. Ward's Dendrogram

![Dendrogram](../figures/fig_S1_dendrogram.png)

*Ward's dendrogram on five MCA factorial coordinates (N = 511). The red line indicates the six-type solution selected on the basis of silhouette and Calinski-Harabasz indices (Table 1, Panel B). The hierarchical structure shows the first split separating Peripheral-Deprived departments from the rest, followed by the metropolitan-educated complex splitting from intermediate types.*

### Figure S2. MCA and Clustering Diagnostics

![Diagnostics](../figures/fig_S2_diagnostics_panel.png)

*Diagnostic panel. (a) Raw MCA eigenvalues; (b) Benzecri-corrected eigenvalues with retention threshold; (c) Silhouette and Calinski-Harabasz indices for k = 3-7; (d) Per-sample silhouette scores for the six-type solution.*

### Figure S3. Longitudinal Rank Stability of ECI_software (2015–2025)

![Temporal panel](../figures/fig_panel_trajectories.png)

*Longitudinal stability of ECI_software across eleven cumulative annual snapshots (2015–2025). (a) Mean ECI_software by departmental type over time, with standard-error bands; the divergence between types is established by 2018 and amplifies thereafter, consistent with path-dependent capability accumulation. (b) Scatter of ECI_software in 2020 versus 2025 among the 125 departments present in both snapshots (Spearman rho = 0.78, *p* < 0.001); the dashed line indicates the 1:1 correspondence. Consecutive-year Spearman rank correlations range from 0.82 to 0.97 across all year pairs (2015–2025); 68 per cent of departments in the top complexity quintile in 2020 remain in the top quintile in 2025 (against 20 per cent expected under random reassignment).*

---

### Table S1. Full ECI Ranking

Complete ranking of 224 departments by ECI_software, with cluster assignment and sociodemographic variables.

**File:** `table_S1_eci_full_ranking.csv`

| Column | Description |
|--------|-------------|
| dpto5 | INDEC department code |
| departamento | Department name |
| provincia | Province |
| region | Region (CABA, Pampeana, NOA, NEA, Cuyo, Patagonia) |
| mca_cluster_label | MCA-CAH cluster assignment |
| eci_software | Economic Complexity Index (standardised) |
| eci_diversity | Number of languages with RCA >= 1 |
| eci_avg_ubiquity | Mean ubiquity of RCA languages |
| gh_total_developers | Total geocoded developers |
| gh_total_repos | Total repositories |
| gh_devs_per_10k | Developers per 10,000 inhabitants |
| pob_2010, pob_2022 | Population (census) |
| pct_jefe_sec_2010 | % household heads with secondary education |
| pct_pc_2010 | % households with computer |
| pct_nbi_2010 | % with unsatisfied basic needs |
| rad_2014 | Mean nighttime radiance |

### Table S2. Cluster x Region Cross-Tabulation

Cross-tabulation of six departmental types with six Argentine regions (chi-squared = 351.5, df = 25, *p* < 0.001, Cramer's *V* = 0.371).

**File:** `table_S2_cluster_region_crosstab.csv`

### Table S3. Small-Type Departments

Individual-level data for departments in Peripheral-Deprived (N = 13 with ECI) and Semi-Rural-Active (N = 5 with ECI), where multivariate regression is not feasible due to small sample size. Bivariate correlations with bootstrap confidence intervals are reported in the article text.

**File:** `table_S3_small_types_data.csv`

### Table S4. Within-Type Correlations with ECI_software

Pearson correlations between ECI_software and 16 candidate variables, computed separately for each departmental type (MCA-CAH classification). Variables were selected from the full set of ~208 available indicators on the basis of theoretical relevance across five domains: education, demography, infrastructure, digital access, and spatial inequality. Only the *N* departments with non-missing ECI values within each type enter the within-type correlations. Correlations for Semi-Rural-Active (*N* = 5) are reported for completeness but should not be interpreted given the minimal sample size.

**File:** `table_S4_within_type_correlations.csv`

| Column | Description |
|--------|-------------|
| variable | Variable name in art1.departamentos |
| domain | Thematic domain |
| national_r / national_n | Pearson *r* with ECI and *N* for the full sample |
| [type]_r / [type]_n | Pearson *r* with ECI and *N* within each departmental type |

### Table S5. Bundle Robustness Check: ECI_individual vs ECI_bundle

Departmental rankings under the individual-language ECI_software and a bundle-based ECI recomputed after mapping the 87 retained languages onto the 38 software-bundle clusters identified by Juhász et al. (2026). Pearson *r* = 0.90 and Spearman rho = 0.90 across all 224 departments. Rank_change = Rank_individual – Rank_bundle; positive values indicate that a department ranks higher under the individual-language measure.

**File:** `table_s_bundle_robustness.csv`

| Column | Description |
|--------|-------------|
| dpto5 | INDEC department code |
| Departamento | Department name |
| Provincia | Province |
| ECI_individual | Individual-language ECI_software (standardised) |
| Rank_individual | Rank under individual-language ECI |
| ECI_bundle | Bundle-based ECI (standardised) |
| Rank_bundle | Rank under bundle ECI |
| Rank_change | Rank_individual – Rank_bundle |

### Table S6. Pairwise Correlations among Predictor Variables

Pearson correlations among the six predictor variables used in the pooled and within-type regressions (log population, secondary education rate, university education rate, computer ownership, nighttime radiance, employment rate), computed for the 224 departments with ECI data. Provided for assessment of multicollinearity; maximum variance inflation factor in the pooled model is 5.25.

**File:** `table_S5_correlation_matrix.csv`
