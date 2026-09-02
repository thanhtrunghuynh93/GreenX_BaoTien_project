# Green-X — Paper contributions & writing plan (AI × environment journal)

## Context

The pipeline is now methodologically sound (leakage-free, seeded, honest metrics, claim scoped to within-system interpolation). The user wants to write a cross-disciplinary AI × environment paper and asked (a) which contributions would lift it above "existing ML applied to environmental data", and (b) for writing that avoids AI slop.

**Assessment:** as-is, the pipeline is the standard applied-ML template (literature-compiled dataset → model zoo → tuning → SHAP) — hundreds of such adsorption papers exist. The distinguishing asset we already have is the **measured generalization gap**: interpolation R² 0.87 vs unseen-study R² ≈ −0.35, with the leakage mechanisms identified and quantified. That is a finding most published adsorption-ML papers are silently subject to.

## Scope (user confirmed)

**Contributions: C1 + C2 + C4. Venue: environment-first** (Water Research / J. Hazard. Mater. / CEJ style): chemistry framing leads, mechanism discussion prominent, ML methodology detail goes to Supplementary Information. C3 and C5 are out of scope (C5 can be a one-line data-availability offer).

## Contributions

**C1 — Generalization audit as the headline (recommended; feasible with current data only)**
- Full leave-one-study-out (LOSO) evaluation over all 59 studies: per-study error distribution, not one split draw.
- Learning curve vs number of training *studies* (does more literature help?).
- Characterize which systems transfer (per-study MAE vs study properties: pollutant class, metal pair, data volume).
- Positions the paper as: "How far do literature-trained adsorption models actually generalize? A leakage-aware evaluation" — AI-side novelty (evaluation rigor, echoes the data-leakage reproducibility literature) × environment-side utility (what these models can/can't be used for).

**C2 — Uncertainty quantification / applicability domain (high practical value, cheap)**
- Conformal prediction intervals (split-conformal on the interpolation task; group-conformal for new studies) with empirical coverage tests.
- Distance-based applicability-domain flag: "trust / don't trust" for a queried system.
- Gives environmental practitioners a usable screening tool rather than a bare point estimate.

**C3 — Study-aware modeling (ML novelty; attacks the negative result)**
- Baselines: global model (current) vs study-intercept/mixed-effects style model vs per-study calibration with k shots (fine-tune on a few rows of a new system).
- Research question: how many measurements of a *new* system are needed before the literature-trained model becomes useful? ("few-shot calibration curve") — a genuinely novel, practical framing.

**C4 — Chemistry-informed features & mechanistic reading (cross-discipline glue)**
- Engineered features with physical meaning: pH − pHpzc (electrostatics), Ci·V/AD-type loading ratios, hydrophobicity interactions.
- SHAP dependence plots read against known adsorption mechanisms; agreement/disagreement discussed as science, not decoration.

**C5 — FAIR release (low effort, reviewers reward it)**
- Publish the curated 59-study dataset + the leakage-aware split protocol + `ldh_pipeline.py` as a reproducible benchmark.

## Implementation outline (code, all reusing `notebooks/ldh_pipeline.py`)

1. Extend `ldh_pipeline.py`:
   - `loso_evaluation()` — leave-one-study-out via GroupKFold(n_splits = n_studies); returns per-study metrics DataFrame.
   - `learning_curve_by_studies()` — cross-study error vs number of training studies (repeated subsampling, seeded).
   - Split-conformal wrapper (hand-rolled, ~40 lines: calibration-set residual quantiles; grouped variant for new-study intervals) + empirical coverage check. Applicability-domain flag: distance of a query row to the training distribution (kNN distance in preprocessed space), correlated against realized error.
   - Chemistry-informed feature block behind `ExperimentConfig` flag `engineered_features`: `pH − pHpzc` (electrostatic driving force), `Ci/AD` loading ratio, `Log Kow × (pH − pHpzc)`-style interaction; compare interpolation and LOSO performance with/without.
2. New notebook `notebooks/generalization_analysis.ipynb` producing the paper's core figures into `Model/paper_figures/`:
   - Fig 1: per-study LOSO error distribution (headline).
   - Fig 2: interpolation vs unseen-study performance + learning curve by #studies.
   - Fig 3: conformal coverage + applicability domain vs realized error.
   - Fig 4: SHAP dependence panels read against adsorption mechanisms (pH−pHpzc, Log Kow, BET), engineered vs raw features.
3. Paper skeleton in `paper/` (environment-first IMRAD: Intro → Methods (data curation, leakage-aware protocol) → Results & Discussion (performance, uncertainty, mechanism) → Environmental implications), using the scientific-writing skill's scaffolds — claims registry + source manifest; every numeric claim traces to a CSV/figure; all literature citations left as `[UNVERIFIED — user to confirm]` placeholders, none invented. ML methodology detail drafted for SI.

## Writing discipline (anti-slop)

- Follow scientific-writing skill: evidence-bound claims (claim/evidence IDs), preserve uncertainty, report the negative cross-study result prominently, no invented citations — literature placeholders marked `[UNVERIFIED]` until the user confirms sources.
- Follow article-writing rules: lead sections with the concrete result/figure, no "rapidly evolving landscape" openers, no "novel framework" adjectives — proof over adjectives; numbers in every claim; limitations stated as bounded facts.
- Frame honestly: not "our model achieves R² 0.87" but "random-split evaluation reports 0.87; unseen-study evaluation reports ≈0; here is what that means for using literature-trained models."

## Verification

- LOSO/conformal/few-shot notebooks execute end-to-end in `.venv` (same nbconvert flow as before).
- Coverage claims checked empirically (conformal coverage ≈ nominal on held-out folds).
- Paper draft passes the skill's lint scripts; every numeric claim traces to a generated CSV/figure in `Model/`.

## Open decisions (asked via AskUserQuestion)

- Which contributions to include (C1 recommended core; C2–C5 selectable).
- Target venue family (environment-first vs AI-for-science) — shapes framing and length.
