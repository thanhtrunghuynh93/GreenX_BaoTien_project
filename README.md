# Green-X Project — LDH Adsorption Modeling

Machine-learning analysis of **micropollutant adsorption on Layered Double Hydroxide (LDH) materials**. Regression models predict adsorption performance — removal rate (**RR %**) and, for the original dataset, adsorption capacity (**Qe mg/g**) — from micropollutant properties, LDH composition and synthesis conditions, material texture, and experimental conditions, followed by feature-importance analysis.

## Scope of the claim

The dataset's 1342 rows come from **59 source studies** (`Id` column = `Author_Year`); within one study the material and pollutant descriptors are constant and only the operating conditions (pH, temperature, time, dosage, Ci) vary. The models are therefore trained and validated to:

> **Predict RR % for *studied* material–pollutant systems at new operating conditions** (within-system interpolation; row-level split with exact feature-duplicate rows removed).

They are **not** validated for screening unseen materials or pollutants. A grouped split by source study — reported in the main notebook as a limitation analysis and saved to `Model/claim_scope_comparison.csv` — shows near-zero cross-study R². Do not cite the interpolation metrics as evidence of screening power.

> **September 2026 rewrite.** A code review found result-invalidating bugs in the previous notebooks (preprocessing fit on the full dataset before splitting, an RR % unit error that squashed the target to [0, 0.01], a "no V" variant that never dropped V, feature selection that never reached the models, and CV metrics mislabeled as train scores — which is why "test" used to look better than "train"). All notebooks were rebuilt on a single shared module, [notebooks/ldh_pipeline.py](notebooks/ldh_pipeline.py), with a leakage-free, fully seeded protocol and the claim scoped as above.

## Project structure

```
├── Data/
│   ├── Data_Adsorption.xlsx            # Original dataset (1343 rows × 30 cols, Qe + RR%, 33 studies)
│   └── 20260804_Data_Adsorption.xlsx   # Updated dataset (1342 rows × 32 cols, RR% only, 59 studies)
├── notebooks/
│   ├── ldh_pipeline.py                 # Shared module: loading, preprocessing, splitting,
│   │                                   # tuning, evaluation, interpretation, experiment runner
│   ├── 20260804_ldh_adsorption_analysis.ipynb   # Main analysis (updated dataset): EDA, VIF,
│   │                                   # primary interpolation experiments + generalization limit
│   ├── Unscaled_no_V_ldh_adsorption_analysis.ipynb  # Variant: V descriptor actually dropped
│   ├── Scaled_ldh_adsorption_analysis.ipynb     # Variant: all features + LightGBM (see note below)
│   └── ldh_adsorption_analysis.ipynb   # Original dataset, both targets (Qe mg/g and RR %)
├── Model/
│   ├── <experiment_name>/              # One folder per experiment, written by run_experiment():
│   │   ├── saved_models.joblib         #   fitted pipelines (accept raw data — preprocessing inside)
│   │   ├── model_scores.csv            #   train / CV-validation / test metrics per model
│   │   ├── permutation_importance.csv  #   feature ranks per model (hold-out test set)
│   │   ├── config.json                 #   the exact experiment config
│   │   └── *.png                       #   comparison, parity, SHAP figures
│   ├── claim_scope_comparison.csv      # Interpolation vs unseen-study test metrics
│   └── Experiment_* / *_42 / ...       # Legacy folders from the pre-rewrite runs (kept for reference;
│                                       # produced by the old, buggy protocol — do not cite)
└── Notes/
    └── Green-X Report Adsorption 17_8.docx
```

Experiments: `interp_row_42` (primary), `corr_drop_interp_42`, `generalization_group_42` (limitation analysis), `no_V_interp_42`, `all_features_lgb_interp_42`, `orig_qe_interp_42`, `orig_rr_interp_42`.

## Data

Each row is one adsorption experiment compiled from the literature. Features fall into four groups:

| Group | Columns |
|---|---|
| Micropollutant descriptors | `TP`, `Log Kow`, `MW g/mol`, Abraham parameters `E, S, A, B, V` (+ `TSPA` in the updated set) |
| LDH composition & synthesis | `Metal(II)`, `Metal(III)`, `M(II)/(III)`, `Method`, `TemH oC`, `HT hour` (+ calcination temp/time and averaged elemental descriptors in the updated set) |
| Material texture | `d(003)`, `BET m2/g`, `PV cm3/g`, `APD nm`, `pHpzc` |
| Adsorption conditions | `pH`, `Tem`, `Time h`, `AD g/L`, `Ci mg/L` |

Targets: `RR %` (normalized to [0, 100] by a range-aware converter — the raw sheets store fractions) and, in the original dataset only, `Qe mg/g`. The `Id` column is excluded from the features and used for the grouped limitation analysis.

## Methodology (shared by all notebooks)

1. **Load & clean** (`ldh_pipeline.load_data`): normalize column names and categorical values, range-aware RR % normalization with an assertion.
2. **Deduplicate & split**: exact feature-duplicate rows are removed (they would let the test be answered by verbatim lookup); primary protocol is a seeded 80/20 row-level split matching the interpolation claim. The limitation analysis uses `GroupShuffleSplit`/`GroupKFold` by source study.
3. **Leakage-free preprocessing inside each model pipeline**: KNN imputation (computed in z-score space, returned in physical units), ordinal encoding of categoricals, StandardScaler for scale-sensitive models (LR, KNN, ANN, Cubist) — all fit on training folds only.
4. **Models**: LR, KNN (grid-searched), ANN/MLP, Cubist (grid-searched), RF / XGBoost / HistGradientBoosting / LightGBM (Optuna, 30 trials, seeded TPE sampler). Tuning and reporting use different fold seeds.
5. **Honest evaluation** (`evaluate_model`): three labeled numbers per model — **train** (in-sample), **CV validation** (mean across folds), **test** (hold-out). With this labeling train ≥ CV ≈ test is the expected ordering.
6. **Interpretation**: permutation importance and SHAP (TreeExplainer where possible) computed with the already-fitted models on the hold-out test set only.
7. **Reproducible artifacts**: every experiment writes models + scores + config + figures to `Model/<experiment_name>/`.

### Why the old results reported "test better than train"

The previous notebooks compared a 5-fold **CV validation** score (models seeing only ~64% of the data) against a hold-out score of a model trained on 80%, while imputation/scaling/feature-selection were fit on all rows before an unseeded split. The rewrite reports the three scores under their real names.

### A note on the variant notebooks

- The **scaled vs unscaled** distinction is retired: scaling now happens inside each model's pipeline where it belongs (tree models are scale-invariant). The old pair never delivered the distinction anyway — the scaled frame was overwritten before training and the "unscaled" one left five imputed columns z-scored.
- The **no V** variant now actually drops V (the old one kept it via an explicit `f != "V"` filter).

## Running locally

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/20260804_ldh_adsorption_analysis.ipynb
```

Paths resolve from the repo layout automatically (`ldh_pipeline.DATA_DIR` / `MODEL_DIR`); on Colab, override those two variables after mounting Drive.

Saved pipelines accept **raw** feature data (preprocessing is embedded):

```python
import joblib
models = joblib.load("Model/interp_row_42/saved_models.joblib")
models["HistGB"].predict(raw_feature_dataframe)
```

Tip: to keep the repo light, strip notebook outputs before committing (`pip install nbstripout && nbstripout --install`).
