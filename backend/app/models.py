"""FR2 (current model rebuild + diagnosis) and FR3 (T-learner uplift + Qini)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import MODEL_VERSION, SEED, current_week
from .db import connect
from .features import FEATURES, build_features, load_raw, to_jsonable, training_frame, visits_last_12m

USUAL_CALL_THRESHOLD = 3  # 3+ visits in last 12 months = "usual" call (and the old validation universe)


# ---------- helpers ----------
def top_decile_precision(y, s) -> float:
    k = max(1, int(len(s) * 0.1))
    idx = np.argsort(-np.asarray(s))[:k]
    return float(np.asarray(y)[idx].mean())


def qini_curve(score, y, t, points: int = 50):
    order = np.argsort(-np.asarray(score))
    y, t = np.asarray(y, float)[order], np.asarray(t, int)[order]
    n = len(y)
    cy_t, cy_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    cn_t, cn_c = np.cumsum(t), np.cumsum(1 - t)
    q = cy_t - np.divide(cy_c * cn_t, cn_c, out=np.zeros(n), where=cn_c > 0)
    q = np.concatenate([[0.0], q])
    x = np.arange(n + 1) / n
    random_line = x * q[-1]
    coef = float(np.trapezoid(q - random_line, x) / max(n, 1))
    idx = np.unique(np.linspace(0, n, points + 1).astype(int))
    return {"x": x[idx].round(4).tolist(), "y": q[idx].round(3).tolist(),
            "random": random_line[idx].round(3).tolist(), "coefficient": round(coef, 5)}


def top_decile_incremental(score, y, t, frac=0.1, n_boot=300, seed=SEED):
    rng = np.random.default_rng(seed)
    order = np.argsort(-np.asarray(score))[: max(10, int(len(score) * frac))]
    yy, tt = np.asarray(y)[order], np.asarray(t)[order]

    def est(a, b):
        if b.sum() == 0 or (1 - b).sum() == 0:
            return np.nan
        return a[b == 1].mean() - a[b == 0].mean()

    point = est(yy, tt)
    boots = [est(yy[i], tt[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(n_boot))]
    boots = np.array([b for b in boots if not np.isnan(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
    return {"estimate": None if np.isnan(point) else round(float(point), 3),
            "ci_low": None if np.isnan(lo) else round(float(lo), 3),
            "ci_high": None if np.isnan(hi) else round(float(hi), 3), "n": int(len(yy))}


def _gbc():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)


def _gbr():
    return GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     min_samples_leaf=20, random_state=SEED)


class TLearner:
    def __init__(self):
        self.m1, self.m0 = _gbr(), _gbr()

    def fit(self, X, y, t, w):
        self.m1.fit(X[t == 1], y[t == 1], sample_weight=w[t == 1])
        self.m0.fit(X[t == 0], y[t == 0], sample_weight=w[t == 0])
        return self

    def predict(self, X):
        # per-quarter uplift -> monthly TRx units to keep interpretation simple
        return (self.m1.predict(X) - self.m0.predict(X)) / 3.0


def ablation_drivers(model: TLearner, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """SHAP-style local attribution: uplift(x) - uplift(x with feature j at population median)."""
    base = model.predict(X.values)
    med = X.median()
    contrib = np.zeros((len(X), len(FEATURES)))
    for j, col in enumerate(FEATURES):
        Xa = X.copy()
        Xa[col] = med[col]
        contrib[:, j] = base - model.predict(Xa.values)
    out = []
    for i in range(len(X)):
        order = np.argsort(-contrib[i])
        picks = [j for j in order if contrib[i, j] > 0][:k]
        if len(picks) < k:
            picks += [j for j in np.argsort(-np.abs(contrib[i])) if j not in picks][: k - len(picks)]
        out.append([{"feature": FEATURES[j], "contribution": round(float(contrib[i, j]), 4),
                     "value": to_jsonable(X.iloc[i])[FEATURES[j]]} for j in picks])
    return out


# ---------- main training / scoring ----------
def train_and_score() -> dict:
    week = current_week()
    with connect() as c:
        raw = load_raw(c)
        df = training_frame(raw)
        rng = np.random.default_rng(SEED)

        # Holdout: 60% of randomized pilot + 30% of the rest
        u = rng.random(len(df))
        holdout = np.where(df.pilot == 1, u < 0.6, u < 0.3)
        tr, te = df[~holdout], df[holdout]
        Xtr, Xte = tr[FEATURES].values, te[FEATURES].values

        # ---- FR2: current model ("will prescribe more next quarter") incl. visit history ----
        cur = _gbc().fit(Xtr, tr.increased.values)
        s_te = cur.predict_proba(Xte)[:, 1]
        val_mask = te.visits_12m.values >= USUAL_CALL_THRESHOLD
        s_all = cur.predict_proba(df[FEATURES].values)[:, 1]
        heavy = df.visits_12m >= df.visits_12m.quantile(0.75)
        top_dec = s_all >= np.quantile(s_all, 0.9)
        current_metrics = {
            "validation_universe": {
                "definition": f"HCPs with {USUAL_CALL_THRESHOLD}+ visits in last 12 months",
                "n": int(val_mask.sum()),
                "auc": round(float(roc_auc_score(te.increased[val_mask], s_te[val_mask])), 4),
                "top_decile_precision": round(top_decile_precision(te.increased[val_mask], s_te[val_mask]), 4),
            },
            "full_universe": {
                "n": int(len(te)),
                "auc": round(float(roc_auc_score(te.increased, s_te)), 4),
                "top_decile_precision": round(top_decile_precision(te.increased, s_te), 4),
            },
            "rarely_visited_universe": {
                "n": int((~val_mask).sum()),
                "auc": round(float(roc_auc_score(te.increased[~val_mask], s_te[~val_mask])), 4)
                if te.increased[~val_mask].nunique() > 1 else None,
            },
            "feature_importance": sorted(
                [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(FEATURES, cur.feature_importances_)],
                key=lambda d: -d["importance"]),
            "top_decile_heavy_visit_overlap_pct": round(float((top_dec & heavy).sum() / top_dec.sum() * 100), 1),
            "base_rate_heavy_visit_pct": round(float(heavy.mean() * 100), 1),
        }

        # ---- FR3: propensity + T-learner ----
        scaler = StandardScaler().fit(Xtr)
        prop = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), tr.treated.values)
        p = np.clip(prop.predict_proba(scaler.transform(Xtr))[:, 1], 0.05, 0.95)
        p = np.where(tr.pilot.values == 1, 0.5, p)
        t = tr.treated.values
        w = np.where(t == 1, 1 / p, 1 / (1 - p))
        tl = TLearner().fit(Xtr, tr.outcome.values.astype(float), t, w)

        up_te = tl.predict(Xte)
        rp = te.pilot.values == 1
        y_rp, t_rp = te.outcome.values[rp], te.treated.values[rp]
        qini_up = qini_curve(up_te[rp], y_rp, t_rp)
        qini_cur = qini_curve(s_te[rp], y_rp, t_rp)
        rand_scores = np.random.default_rng(SEED + 1).random(rp.sum())
        qini_rand = qini_curve(rand_scores, y_rp, t_rp)

        # Ground truth: evaluation only (restricted schema)
        gt = pd.DataFrame(c.execute("SELECT hcp_id, true_uplift FROM restricted.ground_truth").fetchall()).set_index("hcp_id")
        gt_te = gt.true_uplift.reindex(te.index).values
        uplift_metrics = {
            "learner": "T-learner (GBR x2) with inverse-propensity weights",
            "train_n": int(len(tr)), "holdout_n": int(len(te)), "holdout_randomized_n": int(rp.sum()),
            "qini": {"uplift": qini_up, "current": qini_cur, "random": qini_rand},
            "spearman_true_uplift": {
                "uplift_model": round(float(spearmanr(up_te, gt_te).statistic), 4),
                "current_model": round(float(spearmanr(s_te, gt_te).statistic), 4),
            },
            "top_decile_incremental_trx_randomized": {
                "uplift_model": top_decile_incremental(up_te[rp], y_rp, t_rp),
                "current_model": top_decile_incremental(s_te[rp], y_rp, t_rp),
            },
            "propensity_auc": round(float(roc_auc_score(t, prop.predict_proba(scaler.transform(Xtr))[:, 1])), 4),
        }

        # ---- Weekly scoring on latest window (months 7-15) ----
        Xnow = build_features(raw, 15)
        v12_now = visits_last_12m(raw, 15)
        up_now = tl.predict(Xnow.values)
        cur_now = cur.predict_proba(Xnow.values)[:, 1]
        drivers = ablation_drivers(tl, Xnow)

        up_rank = (-up_now).argsort().argsort() + 1
        cur_rank = (-cur_now).argsort().argsort() + 1
        k = int(len(Xnow) * 0.1)
        usual = (v12_now.values >= USUAL_CALL_THRESHOLD)
        scoring_metrics = {
            "week": week,
            "flagged_n": k,
            "current_top_decile_usual_pct": round(float(usual[cur_rank <= k].mean() * 100), 1),
            "uplift_top_decile_usual_pct": round(float(usual[up_rank <= k].mean() * 100), 1),
            "top_decile_overlap_current_vs_uplift_pct": round(
                float(((cur_rank <= k) & (up_rank <= k)).sum() / k * 100), 1),
            "universe_usual_pct": round(float(usual.mean() * 100), 1),
        }

        # ---- Persist ----
        c.execute("DELETE FROM model_run WHERE week = %s", (week,))
        c.execute("DELETE FROM hcp_feature WHERE week = %s", (week,))
        cur_run = c.execute(
            "INSERT INTO model_run (model_type, version, week, metrics_json) VALUES ('current', %s, %s, %s) RETURNING run_id",
            (MODEL_VERSION, week, json.dumps(current_metrics))).fetchone()["run_id"]
        up_run = c.execute(
            "INSERT INTO model_run (model_type, version, week, metrics_json) VALUES ('uplift', %s, %s, %s) RETURNING run_id",
            (MODEL_VERSION, week, json.dumps({**uplift_metrics, "scoring": scoring_metrics}))).fetchone()["run_id"]

        with c.cursor() as cu:
            with cu.copy("COPY hcp_feature (hcp_id, week, features_json) FROM STDIN") as cp:
                for hid, row in Xnow.iterrows():
                    fj = to_jsonable(row)
                    fj["visits_last_12m"] = int(v12_now[hid])
                    cp.write_row((hid, week, json.dumps(fj)))
            with cu.copy("COPY hcp_score (run_id, hcp_id, week, score, uplift, rank, top_drivers_json) FROM STDIN") as cp:
                for i, hid in enumerate(Xnow.index):
                    cp.write_row((up_run, hid, week, float(up_now[i]), float(up_now[i]), int(up_rank[i]),
                                  json.dumps(drivers[i])))
                    cp.write_row((cur_run, hid, week, float(cur_now[i]), None, int(cur_rank[i]), "[]"))
        c.commit()

    return {"week": week, "current_run": cur_run, "uplift_run": up_run,
            "qini_uplift": qini_up["coefficient"], "qini_current": qini_cur["coefficient"],
            "spearman": uplift_metrics["spearman_true_uplift"]}


if __name__ == "__main__":
    print(train_and_score())
