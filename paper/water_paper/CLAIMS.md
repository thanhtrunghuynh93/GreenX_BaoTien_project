# Claims registry — every number in the manuscript traces to a file

| Claim (location) | Value | Source file | Status |
|---|---|---|---|
| 1,342 records, 59 studies (abstract, §1, §2.1) | 1342 / 59 | `Data/20260804_Data_Adsorption.xlsx` via `ldh_pipeline.load_data` | VERIFIED |
| 83 duplicate records removed (§2.1, Text S1) | 83 | dedup mask in `paper_depth_analysis.ipynb` cell 2 | VERIFIED |
| Descriptor summary table (Tab. 1) | see CSV | `Model/paper_figures/descriptor_stats.csv` | VERIFIED |
| Interpolation mean ± SD, 20 splits (abstract, Tab. 2) | HistGB R² 0.850±0.028, MAE 6.3±0.5; XGB 0.842±0.029, 6.3±0.6 | `repeated_interp_summary.csv` | VERIFIED |
| Interpolation baselines (Tab. 2, §3.2) | per-study mean R² 0.43 / MAE 16.7; per-TP 0.18/21.4; global −0.01/24.9 | `repeated_interp_summary.csv` | VERIFIED |
| Dedup costs 2–3 RMSE points vs naive (§3.2) | 7.9 (no dedup, seed 42) vs 10.0–10.4 | legacy `baseline_row_42` vs `interp_row_42/model_scores.csv` | VERIFIED |
| LOSO median per-study MAE (abstract, §3.3) | Cubist 14.1, HistGB 16.4, ANN 18.1, XGB 18.3 | `loso_summary.csv` (all 7 models) | VERIFIED |
| "two to three times interpolation error" (abstract, §3.3) | 14.1/8.0≈1.8 … 16.4/6.3≈2.6 … 18.3/6.3≈2.9 | `loso_summary.csv` + `repeated_interp_summary.csv` | VERIFIED |
| 20 of 59 studies MAE ≤ 6; 9 > 30 (abstract, §3.3) | 20 / 9 | `loso_per_study.csv` (Cubist) | VERIFIED |
| Per-study R² positive 4 of 32 (§3.3) | 4/32 | `loso_per_study.csv` | VERIFIED |
| LOSO baselines: global 24.2, per-TP 25.1 median MAE; ~40% reduction (abstract, §3.3, §4) | 24.2 / 25.1 vs 14.1 | `loso_baselines.csv` | VERIFIED |
| Cubist beats per-TP baseline in 41/59 studies (§3.3) | 41/59 | `loso_head2head.csv` | VERIFIED |
| Transfer by chemistry (§3.3, Fig. transfer): cipro median 5.4 (32 studies), Zn–Al 4.7 (28), tetracycline 20.7, diclofenac 21.8, levofloxacin 22.2, Ni–Fe 30.2; r(n_rows,MAE)=0.45, r(spread,MAE)=0.45 | see CSV | `transfer_breakdown.csv` | VERIFIED |
| Learning curve flat 5→40 studies (§3.3) | 33.8→35.1 median MAE | `learning_curve.csv` | VERIFIED |
| Conformal: interp 0.933 / ±18.8; unseen 0.989 / ±67.6 (abstract, §3.4) | see CSV | `conformal_coverage.csv` | VERIFIED |
| AD flag uninformative both tasks (abstract, §3.4) | r=0.06 grouped, r=0.01 interp | `applicability_domain*.csv` | VERIFIED |
| Engineered features ≈ unchanged (§3.5) | interp MAE 5.62→5.76; LOSO median 16.4→15.96 | `engineered_features_effect.csv` | VERIFIED |
| Permutation ranks table (Tab. 3) | dose/time/Ci/pH lead 6 of 7 models | `perm_ranks_all_models.csv` | VERIFIED |
| PDP shapes: dose saturation ~2–5 g/L, time plateau, Ci decline >~200 mg/L (§3.5) | Fig. S2 | `fig_pdp.png` | VERIFIED (visual) |
| pH−pHpzc threshold penalty ≈+3, up to −9 pts, hydrophilic (§3.5) | Fig. 8a, S3 | `fig_shap_mechanism.png`, `fig_shap_interaction.png` | VERIFIED (visual) |
| Log Kow SHAP −5…+15 monotone (§3.5) | Fig. 8b | `fig_shap_mechanism.png` | VERIFIED (visual) |
| BET conditional attribution positive at low BET; raw medians 43.8 vs 50.0 (§3.5) | computed | raw data check (BET<20: n=225, 0% calcined) | VERIFIED |
| Timing: all fits <1 s except XGB 7.2 s (§3.6, Tab. S3) | see CSV | `computation_time.csv` | VERIFIED |
| Hyperparameters (Tab. S1) | HistGB 484/0.089/14/8; XGB 495/0.114/13/7; RF 347/13/3; Cubist 9/1; KNN k=1 | `Model/interp_row_42/saved_models.joblib` | VERIFIED |
| Missingness: calcination 65.9%, pHpzc 24.6%, PV 25.3% (§3.1) | see CSV | `descriptor_stats.csv` | VERIFIED |

| Spearman vs RR% (Tab. 2): max |ρ|=0.43 (PV), dose 0.33, E −0.30, pH 0.27 | see CSV | `spearman_vs_target.csv` | VERIFIED |
| Target stats: mean 45.5, median 43.7, SD 28.0, skew 0.11 (§3.1) | computed | `paper_depth_analysis.ipynb` A9 output | VERIFIED |
| Composition: 14 pollutants, 13 metal pairs, 4 routes; diclofenac 304 rec/3 studies; cipro 181/32 (§3.1, Tab. S5) | see CSV | `categorical_counts.csv` | VERIFIED |
| VIF: median ≈7×10⁴, max 2.5×10⁶, 22/26 > 10 (§3.1, Tab. S6) | see CSV | `vif_table.csv` | VERIFIED |
| Feature selection (Tab. 4): R² 0.85±0.03 all three sets; LOSO 16.4→16.2 | see CSV | `feature_selection_results.csv` | VERIFIED |
| Top-10 set = 5 conditions + 4 texture + Log Kow (§fs) | ranks | `perm_ranks_all_models.csv` (median rank) | VERIFIED |
| SHAP group shares: conditions 54.2 / texture 17.8 / comp-synth 14.4 / pollutant 13.5 % (§3.5, Tab. S7) | see CSV | `shap_group_share.csv` | VERIFIED |
| Skewness: dose 6.0, Ci 5.0, time 4.9 (§3.1, Tab. S4) | see CSV | `descriptor_stats_extended.csv` | VERIFIED |

Open items before submission:
- Replace 8 remaining `TODO-*` bib keys with verified references; human-verify the 7 assistant-added standard references in `main.bib`.
- Mechanism paragraphs in §3.5 flagged for the authors' domain review.
- Graphical abstract (fig_workflow.png is the basis), author names, CRediT, repository URL.
