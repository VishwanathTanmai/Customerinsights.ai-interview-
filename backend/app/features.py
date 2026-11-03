"""Feature building from raw Postgres tables (no ground truth access)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "rx_last_9m", "rx_trend_pct", "nbrx_last_9m", "visits_last_9m", "samples_last_9m",
    "months_since_last_visit", "email_open_rate_pct", "panel_size_proxy", "is_derm", "is_np_pa",
]

FEATURE_LABELS = {
    "rx_last_9m": "TRx in last 9 months",
    "rx_trend_pct": "TRx trend, last 3 vs prior 3 months (%)",
    "nbrx_last_9m": "New-to-brand Rx in last 9 months",
    "visits_last_9m": "Rep visits in last 9 months",
    "samples_last_9m": "Samples delivered in last 9 months",
    "months_since_last_visit": "Months since last rep visit",
    "email_open_rate_pct": "Email open rate (%)",
    "panel_size_proxy": "Patient-volume proxy",
    "is_derm": "Dermatologist",
    "is_np_pa": "Derm NP/PA",
}


def load_raw(c) -> dict[str, pd.DataFrame]:
    q = lambda s: pd.DataFrame(c.execute(s).fetchall())
    return {
        "hcp": q("SELECT hcp_id, specialty, territory_id, region, rep_id, panel_size_proxy FROM hcp ORDER BY hcp_id"),
        "rx": q("SELECT hcp_id, period, trx, nbrx FROM rx_history"),
        "calls": q("SELECT hcp_id, period, samples FROM call_activity"),
        "email": q("SELECT hcp_id, period, sent, opened FROM email_engagement"),
        "pilot": q("SELECT hcp_id FROM random_pilot"),
    }


def build_features(raw: dict[str, pd.DataFrame], end: int) -> pd.DataFrame:
    start = end - 8
    h = raw["hcp"].set_index("hcp_id")
    rx = raw["rx"]
    calls = raw["calls"]
    em = raw["email"]

    w = rx[(rx.period >= start) & (rx.period <= end)]
    rx9 = w.groupby("hcp_id").trx.sum()
    nb9 = w.groupby("hcp_id").nbrx.sum()
    recent = rx[(rx.period > end - 3) & (rx.period <= end)].groupby("hcp_id").trx.sum()
    prior = rx[(rx.period > end - 6) & (rx.period <= end - 3)].groupby("hcp_id").trx.sum()

    cw = calls[(calls.period >= start) & (calls.period <= end)]
    v9 = cw.groupby("hcp_id").size()
    s9 = cw.groupby("hcp_id").samples.sum()
    last = calls[calls.period <= end].groupby("hcp_id").period.max()

    ew = em[(em.period >= start) & (em.period <= end)].groupby("hcp_id")[["sent", "opened"]].sum()

    f = pd.DataFrame(index=h.index)
    f["rx_last_9m"] = rx9.reindex(f.index).fillna(0)
    f["rx_trend_pct"] = ((recent.reindex(f.index).fillna(0) - prior.reindex(f.index).fillna(0))
                         / prior.reindex(f.index).fillna(0).clip(lower=1) * 100).round(1)
    f["nbrx_last_9m"] = nb9.reindex(f.index).fillna(0)
    f["visits_last_9m"] = v9.reindex(f.index).fillna(0)
    f["samples_last_9m"] = s9.reindex(f.index).fillna(0)
    f["months_since_last_visit"] = (end - last.reindex(f.index)).fillna(12).clip(upper=12)
    f["email_open_rate_pct"] = (ew.opened / ew.sent.clip(lower=1) * 100).reindex(f.index).fillna(0).round(1)
    f["panel_size_proxy"] = h.panel_size_proxy
    f["is_derm"] = (h.specialty == "Dermatology").astype(int)
    f["is_np_pa"] = (h.specialty == "Derm NP/PA").astype(int)
    return f[FEATURES].astype(float)


def training_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Features at month 9, treatment = visited in months 10-12, outcome = TRx(13-15) - TRx(7-9)."""
    f = build_features(raw, 9)
    rx, calls = raw["rx"], raw["calls"]
    post = rx[rx.period.between(13, 15)].groupby("hcp_id").trx.sum()
    pre = rx[rx.period.between(7, 9)].groupby("hcp_id").trx.sum()
    treated = calls[calls.period.between(10, 12)].groupby("hcp_id").size()
    v12 = calls[calls.period.between(1, 12)].groupby("hcp_id").size()
    f["treated"] = (treated.reindex(f.index).fillna(0) > 0).astype(int)
    f["outcome"] = (post.reindex(f.index).fillna(0) - pre.reindex(f.index).fillna(0))
    f["increased"] = (f.outcome > 0).astype(int)
    f["visits_12m"] = v12.reindex(f.index).fillna(0)
    f["pilot"] = f.index.isin(set(raw["pilot"].hcp_id)).astype(int)
    return f


def visits_last_12m(raw, end: int) -> pd.Series:
    c = raw["calls"]
    s = c[c.period.between(end - 11, end)].groupby("hcp_id").size()
    return s.reindex(raw["hcp"].hcp_id).fillna(0)


def to_jsonable(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        v = float(v)
        out[k] = int(v) if v.is_integer() else round(v, 2)
    return out


__all__ = ["FEATURES", "FEATURE_LABELS", "load_raw", "build_features", "training_frame",
           "visits_last_12m", "to_jsonable", "np"]
