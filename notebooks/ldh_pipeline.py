"""Shared, leakage-free modeling pipeline for the Green-X LDH adsorption study.

All four analysis notebooks import from this module so that fixes apply
everywhere at once (previously each notebook carried its own diverging copy
of this code).

Key guarantees:
- Every fitted transform (imputation, scaling, encoding) lives inside an
  sklearn Pipeline and is fit on training data only.
- Train/test splitting is seeded and, by default, grouped by source study
  (`Id` column) so that rows from one paper never appear on both sides.
- `evaluate_model` reports three honestly-labeled numbers: train score,
  CV-validation score, and hold-out test score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

# Paths resolve from the repo layout; on Colab override DATA_DIR / MODEL_DIR.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"
MODEL_DIR = REPO_ROOT / "Model"

TARGET_CANDIDATES = ["RR %", "Qe mg/g", "Qt", "Ce mg/L"]

METHOD_MAP = {
    "urea hydrolysis": "Urea Hydrolysis",
    "co-precipitation or aging": "Co-precipitation",
    "co-precipitation": "Co-precipitation",
    "co- precipitation": "Co-precipitation",
    "coprecipitation": "Co-precipitation",
    "reconstruction or memory effect": "Reconstruction",
}


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------

def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load and clean the adsorption dataset. Keeps the `Id` (source study)
    column — it is needed for group-aware splitting."""
    df = pd.read_excel(file_path, sheet_name=0, header=1)
    df.columns = df.columns.str.strip()

    for col in ("TP", "Metal(II)", "Metal(III)"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "Metal(II)" in df.columns:
        df["Metal(II)"] = df["Metal(II)"].str.title()

    if "Method" in df.columns:
        # Normalize whitespace/punctuation first, then map known synonyms.
        method = (
            df["Method"].astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"\s*-\s*", "-", regex=True)
            .str.strip(" .")
        )
        df["Method"] = method.replace(METHOD_MAP).str.title()

    df["RR %"] = normalize_rr_percent(df["RR %"])
    return df


def normalize_rr_percent(rr: pd.Series) -> pd.Series:
    """Bring RR to percent units [0, 100] regardless of stored format.

    The raw sheets store RR as a fraction (0-1); older notebook revisions
    divided by 100 unconditionally, squashing the target to [0, 0.01].
    This version is range-aware and idempotent under re-runs.
    """
    rr = rr.astype(float)
    if rr.max() <= 1.5:  # stored as fraction
        rr = rr * 100
    assert rr.min() >= 0 and rr.max() <= 100 + 1e-9, (
        f"RR % out of range after normalization: [{rr.min()}, {rr.max()}]"
    )
    return rr


# ---------------------------------------------------------------------------
# Preprocessing (all inside the model pipeline — fit on train only)
# ---------------------------------------------------------------------------

class ScaledKNNImputer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """KNN imputation computed in z-score space, returned in original units.

    KNN distances need comparable scales, but downstream consumers may want
    physical units, so we scale -> impute -> inverse-transform.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        # A column with no observed values (possible in small grouped folds)
        # yields NaN statistics; neutralize so transform stays finite, and let
        # KNNImputer keep the empty feature (imputed at the z-space mean, 0).
        self.scaler_.mean_ = np.nan_to_num(self.scaler_.mean_, nan=0.0)
        self.scaler_.scale_ = np.where(np.isnan(self.scaler_.scale_), 1.0,
                                       self.scaler_.scale_)
        self.imputer_ = KNNImputer(
            n_neighbors=self.n_neighbors, keep_empty_features=True
        ).fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        imputed = self.imputer_.transform(self.scaler_.transform(X))
        return self.scaler_.inverse_transform(imputed)


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", ScaledKNNImputer(), numeric_cols),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


# ---------------------------------------------------------------------------
# Splitting (grouped by source study to prevent same-study leakage)
# ---------------------------------------------------------------------------

def make_split(X, y, groups=None, test_size=0.2, seed=SEED):
    """Grouped split when `groups` is given, plain seeded split otherwise."""
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        assert set(np.asarray(groups)[tr_idx]).isdisjoint(np.asarray(groups)[te_idx])
        return (
            X.iloc[tr_idx], X.iloc[te_idx],
            y.iloc[tr_idx], y.iloc[te_idx],
            np.asarray(groups)[tr_idx],
        )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_tr, X_te, y_tr, y_te, None


def make_cv(train_groups=None, n_splits=5, seed=SEED):
    """CV iterator for tuning/validation. GroupKFold when groups are known."""
    if train_groups is not None:
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Models & tuning
# ---------------------------------------------------------------------------

# Optuna search spaces: {param: (kind, low, high, extra_kwargs)}
OPTUNA_SPACES = {
    "HistGB": (
        HistGradientBoostingRegressor,
        {
            "max_iter": ("int", 100, 600, {}),
            "learning_rate": ("float", 1e-4, 1.0, {"log": True}),
            "max_depth": ("int", 3, 15, {}),
            "min_samples_leaf": ("int", 3, 9, {}),
        },
    ),
    "RF": (
        RandomForestRegressor,
        {
            "n_estimators": ("int", 100, 600, {}),
            "max_depth": ("int", 3, 15, {}),
            "min_samples_leaf": ("int", 3, 9, {}),
        },
    ),
}
try:  # optional heavy deps
    from xgboost import XGBRegressor

    OPTUNA_SPACES["XGB"] = (
        XGBRegressor,
        {
            "n_estimators": ("int", 100, 600, {}),
            "learning_rate": ("float", 1e-4, 1.0, {"log": True}),
            "max_depth": ("int", 3, 15, {}),
            "min_child_weight": ("int", 3, 9, {}),
        },
    )
except ImportError:
    pass
try:
    from lightgbm import LGBMRegressor

    OPTUNA_SPACES["LGB"] = (
        LGBMRegressor,
        {
            "n_estimators": ("int", 100, 600, {}),
            "learning_rate": ("float", 1e-4, 1.0, {"log": True}),
            "max_depth": ("int", 3, 15, {}),
            "min_child_samples": ("int", 3, 9, {}),
        },
    )
except ImportError:
    pass


def _suggest(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, (kind, low, high, kw) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, low, high, **kw)
        else:
            params[name] = trial.suggest_float(name, low, high, **kw)
    return params


def tune_optuna(model_name, preprocessor, X, y, cv, groups=None, n_trials=30, seed=SEED):
    """Single dict-driven tuner for all tree ensembles.

    The seeded TPESampler IS passed to create_study (a bug in previous
    notebook copies created it and threw it away).
    """
    estimator_cls, space = OPTUNA_SPACES[model_name]

    def objective(trial):
        extra = {"n_jobs": -1} if model_name in ("XGB", "RF", "LGB") else {}
        if model_name == "LGB":
            extra["verbose"] = -1
        model = make_model_pipeline(
            preprocessor, estimator_cls(**_suggest(trial, space), random_state=seed, **extra)
        )
        scores = _cross_validate_scores(model, X, y, cv, groups)
        return scores["mae"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    extra = {"n_jobs": -1} if model_name in ("XGB", "RF", "LGB") else {}
    if model_name == "LGB":
        extra["verbose"] = -1
    return make_model_pipeline(
        preprocessor, estimator_cls(**study.best_params, random_state=seed, **extra)
    ), study.best_params


NEEDS_SCALER = {"LR", "KNN", "ANN", "Cubist"}


def make_model_pipeline(preprocessor, regressor, with_scaler=False):
    steps = [("prep", clone(preprocessor))]
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", regressor))
    return Pipeline(steps)


def build_models(preprocessor, X, y, cv, groups=None, n_trials=30, seed=SEED,
                 include=("LR", "KNN", "ANN", "Cubist", "RF", "XGB", "HistGB", "LGB")):
    """Return {name: unfitted pipeline or GridSearchCV} for the model zoo.

    Tuning uses folds derived from `cv`/`groups` on the TRAINING data only.
    """
    models: dict[str, object] = {}

    if "LR" in include:
        models["LR"] = make_model_pipeline(preprocessor, LinearRegression(), with_scaler=True)

    if "KNN" in include:
        knn = make_model_pipeline(preprocessor, KNeighborsRegressor(), with_scaler=True)
        models["KNN"] = GridSearchCV(
            knn,
            {"regressor__n_neighbors": list(range(1, 10))},
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
        )

    if "ANN" in include:
        models["ANN"] = make_model_pipeline(
            preprocessor,
            MLPRegressor(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                         max_iter=2000, random_state=seed),
            with_scaler=True,
        )

    if "Cubist" in include:
        try:
            from cubist import Cubist

            cub = make_model_pipeline(preprocessor, Cubist(), with_scaler=True)
            models["Cubist"] = GridSearchCV(
                cub,
                {"regressor__n_committees": list(range(1, 10)),
                 "regressor__neighbors": [None] + list(range(1, 9))},
                cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
            )
        except ImportError:
            print("[warn] cubist not installed — skipping Cubist")

    for name in ("RF", "XGB", "HistGB", "LGB"):
        if name in include and name in OPTUNA_SPACES:
            print(f"--- Optuna tuning {name} ({n_trials} trials) ---")
            model, best = tune_optuna(name, preprocessor, X, y, cv, groups,
                                      n_trials=n_trials, seed=seed)
            print(f"[*] {name} best params: {best}")
            models[name] = model

    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cross_validate_scores(model, X, y, cv, groups=None):
    res = cross_validate(
        model, X, y, cv=cv, groups=groups,
        scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error",
                 "neg_rmse": "neg_root_mean_squared_error"},
        n_jobs=-1,
    )
    return {
        "r2": np.mean(res["test_r2"]),
        "mae": -np.mean(res["test_neg_mae"]),
        "rmse": -np.mean(res["test_neg_rmse"]),
    }


def evaluate_model(model, X_tr, y_tr, X_te, y_te, cv, model_name, groups=None):
    """Fit + report three honestly-labeled scores.

    - train:   fit on the full training set, scored on that same data
               (an optimistic in-sample ceiling).
    - cv_val:  mean 5-fold validation score across folds of the training set
               (each fold's model sees only 4/5 of the training data — this
               is a *validation* score, not a train score).
    - test:    the train-fitted model scored on the untouched hold-out.

    With this labeling, train >= test is the expected ordering and cv_val
    typically sits near test (slightly below when data is scarce).
    """
    print(f"--- {model_name} ---")

    if isinstance(model, GridSearchCV):
        model.fit(X_tr, y_tr, groups=groups) if groups is not None else model.fit(X_tr, y_tr)
        best_model = model.best_estimator_  # already refit on full train
        clean = {k.split("__")[-1]: v for k, v in model.best_params_.items()}
        print(f"[*] best params: {clean}")
    else:
        best_model = model
        best_model.fit(X_tr, y_tr)

    cv_scores = _cross_validate_scores(best_model, X_tr, y_tr, cv, groups)

    y_pred_tr = best_model.predict(X_tr)
    y_pred = best_model.predict(X_te)
    res = {
        "model": best_model, "label": model_name,
        "y_te": np.asarray(y_te), "y_pred": y_pred,
        "train_r2": r2_score(y_tr, y_pred_tr),
        "train_mae": mean_absolute_error(y_tr, y_pred_tr),
        "cv_r2": cv_scores["r2"], "cv_mae": cv_scores["mae"], "cv_rmse": cv_scores["rmse"],
        "test_r2": r2_score(y_te, y_pred),
        "test_mae": mean_absolute_error(y_te, y_pred),
        "test_rmse": root_mean_squared_error(y_te, y_pred),
    }
    print(f"    Train      R2 {res['train_r2']:7.4f} | MAE {res['train_mae']:8.4f}")
    print(f"    CV (valid) R2 {res['cv_r2']:7.4f} | MAE {res['cv_mae']:8.4f}")
    print(f"    Test       R2 {res['test_r2']:7.4f} | MAE {res['test_mae']:8.4f}\n")
    return res


def results_table(results: dict) -> pd.DataFrame:
    rows = {
        name: {k: r[k] for k in
               ("train_r2", "cv_r2", "test_r2", "train_mae", "cv_mae", "test_mae", "test_rmse")}
        for name, r in results.items()
    }
    return pd.DataFrame(rows).T.sort_values("test_mae")


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def permutation_ranks(results: dict, X_te, y_te, n_repeats=10, seed=SEED) -> pd.DataFrame:
    """Permutation importance of each ALREADY-FITTED model on the hold-out
    set (no refitting, no train rows). Returns feature ranks per model."""
    ranks = {}
    for name, res in results.items():
        imp = permutation_importance(
            res["model"], X_te, y_te, n_repeats=n_repeats, random_state=seed, n_jobs=-1
        )
        s = pd.Series(imp.importances_mean, index=X_te.columns)
        ranks[name] = s.rank(ascending=False).astype(int)
    return pd.DataFrame(ranks)


def shap_summary(res: dict, X_te: pd.DataFrame, max_background=200, seed=SEED):
    """SHAP values for one fitted pipeline, computed on the hold-out set.

    Uses TreeExplainer on the transformed space for tree models (exact and
    fast); falls back to the permutation explainer with a small background.
    """
    import shap

    pipe = res["model"]
    prep = Pipeline(pipe.steps[:-1])
    regressor = pipe.steps[-1][1]
    X_te_t = prep.transform(X_te)

    try:
        explainer = shap.TreeExplainer(regressor)
        return explainer(np.asarray(X_te_t, dtype=float)), list(X_te_t.columns)
    except Exception:
        background = shap.sample(np.asarray(X_te_t, dtype=float),
                                 min(len(X_te_t), max_background), random_state=seed)
        explainer = shap.Explainer(regressor.predict, background)
        return explainer(np.asarray(X_te_t, dtype=float)), list(X_te_t.columns)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF over FEATURES ONLY (targets must not be in X)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    bad = [c for c in X.columns if c in TARGET_CANDIDATES]
    assert not bad, f"targets {bad} must be excluded from VIF"
    Xn = X.select_dtypes(include=[np.number]).dropna()
    Xn = Xn.assign(_const=1.0)
    vifs = [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1] - 1)]
    return (pd.DataFrame({"feature": Xn.columns[:-1], "VIF": vifs})
            .sort_values("VIF", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Generalization analyses (paper: C1 audit, C2 UQ/applicability, C4 features)
# ---------------------------------------------------------------------------

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Chemistry-informed features. pH − pHpzc is the electrostatic driving
    force (surface charge sign relative to solution pH); Ci/AD is the initial
    loading per adsorbent mass; the Log Kow interaction couples hydrophobicity
    with surface charge."""
    X = X.copy()
    if {"pH", "pHpzc"} <= set(X.columns):
        X["pH-pHpzc"] = X["pH"] - X["pHpzc"]
    if {"Ci mg/L", "AD g/L"} <= set(X.columns):
        X["Ci/AD"] = X["Ci mg/L"] / X["AD g/L"].replace(0, np.nan)
    if {"Log Kow", "pH", "pHpzc"} <= set(X.columns):
        X["LogKow*(pH-pHpzc)"] = X["Log Kow"] * (X["pH"] - X["pHpzc"])
    return X


def loso_evaluation(models: dict, X: pd.DataFrame, y: pd.Series, groups) -> pd.DataFrame:
    """Leave-one-study-out: refit each model (fixed hyperparameters) on all
    other studies, predict the held-out study. Returns one row per
    (study, model) with MAE/RMSE and the study's row count."""
    groups = np.asarray(groups)
    rows = []
    for tr_idx, te_idx in LeaveOneGroupOut().split(X, y, groups):
        study = groups[te_idx][0]
        y_te = np.asarray(y.iloc[te_idx], dtype=float)
        for name, model in models.items():
            m = clone(model)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = m.predict(X.iloc[te_idx])
            rows.append({
                "study": study, "model": name, "n_rows": len(te_idx),
                "mae": mean_absolute_error(y_te, pred),
                "rmse": root_mean_squared_error(y_te, pred),
                "r2": r2_score(y_te, pred) if len(te_idx) >= 3 and np.var(y_te) > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def learning_curve_by_studies(model, X, y, groups, sizes=(5, 10, 20, 30, 40),
                              n_repeats=10, test_frac=0.2, seed=SEED) -> pd.DataFrame:
    """Cross-study error vs number of training studies. A fixed set of test
    studies is held out; for each size, `n_repeats` random subsets of the
    remaining studies train a clone of `model`."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    studies = np.unique(groups)
    test_studies = rng.choice(studies, size=max(2, int(len(studies) * test_frac)),
                              replace=False)
    te_mask = np.isin(groups, test_studies)
    pool = [s for s in studies if s not in set(test_studies)]
    rows = []
    for n in sizes:
        if n > len(pool):
            continue
        for rep in range(n_repeats):
            chosen = rng.choice(pool, size=n, replace=False)
            tr_mask = np.isin(groups, chosen)
            m = clone(model)
            m.fit(X[tr_mask], y[tr_mask])
            pred = m.predict(X[te_mask])
            rows.append({"n_studies": n, "repeat": rep,
                         "mae": mean_absolute_error(y[te_mask], pred),
                         "rmse": root_mean_squared_error(y[te_mask], pred)})
    return pd.DataFrame(rows)


def repeated_interpolation(models: dict, X, y, n_repeats=20, test_size=0.2,
                           seed=SEED) -> pd.DataFrame:
    """Task (T1) over `n_repeats` seeded record-level splits with fixed
    hyperparameters. Returns one row per (model, repeat); summarize as
    mean +/- SD to remove single-draw luck from reported scores."""
    rows = []
    for rep in range(n_repeats):
        X_tr, X_te, y_tr, y_te, _ = make_split(X, y, groups=None,
                                               test_size=test_size, seed=seed + rep)
        for name, model in models.items():
            m = clone(model)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            rows.append({"model": name, "repeat": rep,
                         "r2": r2_score(y_te, pred),
                         "mae": mean_absolute_error(y_te, pred),
                         "rmse": root_mean_squared_error(y_te, pred)})
    return pd.DataFrame(rows)


def _group_mean_predict(tr_keys, y_tr, te_keys) -> np.ndarray:
    means = pd.Series(np.asarray(y_tr, dtype=float)).groupby(
        np.asarray(tr_keys)).mean()
    pred = pd.Series(np.asarray(te_keys)).map(means)
    return pred.fillna(float(np.mean(y_tr))).to_numpy()


def baseline_repeated_interpolation(y, keys: dict, n_repeats=20, test_size=0.2,
                                    seed=SEED) -> pd.DataFrame:
    """Naive baselines under task (T1): 'global mean' plus one group-mean
    predictor per entry in `keys` (name -> label series aligned with y),
    e.g. per-pollutant or per-study means learned on the training rows."""
    y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    keys = {k: pd.Series(np.asarray(v)).reset_index(drop=True) for k, v in keys.items()}
    rows = []
    idx = np.arange(len(y))
    for rep in range(n_repeats):
        tr_idx, te_idx = train_test_split(idx, test_size=test_size,
                                          random_state=seed + rep)
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        preds = {"global mean": np.full(len(te_idx), y_tr.mean())}
        for name, k in keys.items():
            preds[name] = _group_mean_predict(k.iloc[tr_idx], y_tr, k.iloc[te_idx])
        for name, pred in preds.items():
            rows.append({"model": name, "repeat": rep,
                         "r2": r2_score(y_te, pred),
                         "mae": mean_absolute_error(y_te, pred),
                         "rmse": root_mean_squared_error(y_te, pred)})
    return pd.DataFrame(rows)


def baseline_loso(y, groups, keys: dict) -> pd.DataFrame:
    """Naive baselines under leave-one-study-out: for each held-out study,
    predict the global mean or the group mean (per entry in `keys`) learned
    from the remaining studies."""
    y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    groups = np.asarray(groups)
    keys = {k: pd.Series(np.asarray(v)).reset_index(drop=True) for k, v in keys.items()}
    rows = []
    for study in np.unique(groups):
        te = groups == study
        y_tr, y_te = y[~te], y[te]
        preds = {"global mean": np.full(int(te.sum()), y_tr.mean())}
        for name, k in keys.items():
            preds[name] = _group_mean_predict(k[~te], y_tr, k[te])
        for name, pred in preds.items():
            rows.append({"study": study, "model": name, "n_rows": int(te.sum()),
                         "mae": mean_absolute_error(y_te, pred),
                         "rmse": root_mean_squared_error(y_te, pred)})
    return pd.DataFrame(rows)


def split_conformal(model, X_tr, y_tr, X_te, y_te, alpha=0.1,
                    groups_tr=None, seed=SEED) -> dict:
    """Split-conformal prediction intervals.

    Calibration rows are split off the training set — grouped by study when
    `groups_tr` is given, so calibration residuals come from studies the model
    was not fit on (matching the new-study prediction task). Returns the
    interval half-width q and the empirical test coverage.
    """
    if groups_tr is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        fit_idx, cal_idx = next(gss.split(X_tr, y_tr, groups=groups_tr))
    else:
        idx = np.arange(len(X_tr))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_cal = int(0.25 * len(idx))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]

    m = clone(model)
    m.fit(X_tr.iloc[fit_idx], y_tr.iloc[fit_idx])
    cal_resid = np.abs(np.asarray(y_tr.iloc[cal_idx], dtype=float)
                       - m.predict(X_tr.iloc[cal_idx]))
    n = len(cal_resid)
    q = np.quantile(cal_resid, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                    method="higher")
    pred = m.predict(X_te)
    y_te = np.asarray(y_te, dtype=float)
    covered = (y_te >= pred - q) & (y_te <= pred + q)
    return {"alpha": alpha, "q": float(q), "coverage": float(covered.mean()),
            "mean_width": float(2 * q), "n_calibration": n,
            "pred": pred, "covered": covered, "model": m}


def applicability_domain(fitted_pipeline, X_tr, X_te, k=5) -> np.ndarray:
    """Mean z-scored kNN distance of each test row to the training set, in the
    pipeline's preprocessed feature space. Larger = further outside the
    training distribution."""
    from sklearn.neighbors import NearestNeighbors

    prep = Pipeline(fitted_pipeline.steps[:-1])
    Z_tr = np.asarray(prep.transform(X_tr), dtype=float)
    Z_te = np.asarray(prep.transform(X_te), dtype=float)
    scaler = StandardScaler().fit(Z_tr)
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(Z_tr))
    dist, _ = nn.kneighbors(scaler.transform(Z_te))
    return dist.mean(axis=1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_parity(res: dict, target_label: str, out_path=None, color="#1565C0"):
    """Actual-vs-predicted + residuals on the hold-out set, annotated with
    TEST metrics (previous versions annotated CV metrics over test data)."""
    y_t, y_p = res["y_te"], res["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{res['label']} – {target_label} (hold-out test)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_t, y_p, alpha=0.5, s=30, color=color)
    lo, hi = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(f"Actual vs Predicted\nTest R²={res['test_r2']:.4f}")
    ax.legend()

    ax = axes[1]
    ax.scatter(y_p, y_t - y_p, alpha=0.45, s=30, color=color)
    ax.axhline(0, color="black", ls="--", lw=1.5)
    ax.set_xlabel(f"Predicted {target_label}")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot\nTest MAE={res['test_mae']:.4f}")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_comparison(table: pd.DataFrame, target_label: str, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    table[["train_r2", "cv_r2", "test_r2"]].plot.bar(ax=axes[0], rot=0)
    axes[0].set_title(f"R² – {target_label}")
    axes[0].legend(["Train (in-sample)", "CV validation", "Test (hold-out)"])
    table[["train_mae", "cv_mae", "test_mae"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set_title(f"MAE – {target_label}")
    axes[1].legend(["Train (in-sample)", "CV validation", "Test (hold-out)"])
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    data_file: str = "20260804_Data_Adsorption.xlsx"
    target: str = "RR %"
    drop_features: list[str] = field(default_factory=list)
    split_mode: str = "group"  # "row" = within-system interpolation; "group" = unseen studies
    dedup_features: bool = False  # drop exact feature-duplicate rows before splitting
    engineered_features: bool = False  # add chemistry-informed features (pH-pHpzc, Ci/AD, ...)
    test_size: float = 0.2
    seed: int = SEED
    n_trials: int = 30
    models: tuple = ("LR", "KNN", "ANN", "Cubist", "RF", "XGB", "HistGB")


def run_experiment(cfg: ExperimentConfig, df: pd.DataFrame | None = None) -> dict:
    """End-to-end: split -> tune -> evaluate -> save under MODEL_DIR/<name>/."""
    if df is None:
        df = load_data(DATA_DIR / cfg.data_file)

    out_dir = MODEL_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df[df[cfg.target].notna()].reset_index(drop=True)
    groups_all = df["Id"] if (cfg.split_mode == "group" and "Id" in df.columns) else None
    targets_present = [c for c in TARGET_CANDIDATES if c in df.columns]
    X = df.drop(columns=["Id", *targets_present], errors="ignore")
    X = X.drop(columns=[c for c in cfg.drop_features if c in X.columns])
    if cfg.engineered_features:
        X = add_engineered_features(X)
    y = df[cfg.target]

    for f in cfg.drop_features:
        assert f not in X.columns, f"{f} should have been dropped"

    if cfg.dedup_features:
        # Rows identical in every feature would let the test be answered by
        # verbatim lookup; keep only the first occurrence.
        keep = ~X.duplicated(keep="first")
        print(f"[{cfg.name}] dropped {(~keep).sum()} exact feature-duplicate rows")
        X, y = X[keep], y[keep]
        if groups_all is not None:
            groups_all = groups_all[keep]

    X_tr, X_te, y_tr, y_te, tr_groups = make_split(
        X, y, groups=groups_all, test_size=cfg.test_size, seed=cfg.seed
    )
    print(f"[{cfg.name}] split={cfg.split_mode}  train={len(X_tr)}  test={len(X_te)}  "
          f"features={X.shape[1]}")

    numeric, categorical = split_feature_types(X)
    prep = make_preprocessor(numeric, categorical)
    cv = make_cv(tr_groups, seed=cfg.seed)
    # Reporting CV uses different fold boundaries than tuning CV (M3).
    report_cv = make_cv(tr_groups, seed=cfg.seed + 1)

    models = build_models(prep, X_tr, y_tr, cv, groups=tr_groups,
                          n_trials=cfg.n_trials, seed=cfg.seed, include=cfg.models)

    results = {
        name: evaluate_model(m, X_tr, y_tr, X_te, y_te, report_cv, name, groups=tr_groups)
        for name, m in models.items()
    }

    table = results_table(results)
    table.to_csv(out_dir / "model_scores.csv")
    joblib.dump({n: r["model"] for n, r in results.items()}, out_dir / "saved_models.joblib")
    with open(out_dir / "config.json", "w") as f:
        json.dump({**cfg.__dict__, "models": list(cfg.models)}, f, indent=2, default=str)

    return {"config": cfg, "results": results, "table": table,
            "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
            "out_dir": out_dir}
