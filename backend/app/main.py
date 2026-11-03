"""FastAPI service (PRD section 10). Read API for the UI + feedback/event writes."""
from __future__ import annotations

import threading
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ADMIN_TOKEN, CORS_ORIGINS, current_week
from .db import conn, init_schema
from .features import FEATURE_LABELS

app = FastAPI(title="Uplift HCP Targeting API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-Admin-Token"])


@app.on_event("startup")
def _startup():
    init_schema()


def _latest_week(c, week: str | None) -> str | None:
    if week:
        return week
    r = c.execute("SELECT week FROM model_run WHERE model_type='uplift' ORDER BY run_id DESC LIMIT 1").fetchone()
    return r["week"] if r else None


def _run_id(c, week, model_type="uplift"):
    r = c.execute("SELECT run_id FROM model_run WHERE model_type=%s AND week=%s ORDER BY run_id DESC LIMIT 1",
                  (model_type, week)).fetchone()
    return r["run_id"] if r else None


@app.get("/health")
def health():
    with conn() as c:
        c.execute("SELECT 1")
    return {"ok": True, "week": current_week()}


@app.get("/hcps")
def list_hcps(territory: str | None = None, region: str | None = None, specialty: str | None = None,
              rep: str | None = None, week: str | None = None, q: str | None = None,
              sort: Literal["uplift", "recency"] = "uplift", flagged_only: bool = True,
              limit: int = Query(50, le=500), offset: int = Query(0, ge=0)):
    with conn() as c:
        week = _latest_week(c, week)
        if not week:
            return {"week": None, "total": 0, "items": []}
        run_id = _run_id(c, week)
        n = c.execute("SELECT count(*) AS n FROM hcp_score WHERE run_id=%s", (run_id,)).fetchone()["n"]
        flag_cut = max(1, int(n * 0.1))
        where, args = ["s.run_id = %s"], [run_id]
        if flagged_only:
            where.append("s.rank <= %s"); args.append(flag_cut)
        for col, val in (("h.territory_id", territory), ("h.region", region), ("h.specialty", specialty), ("h.rep_id", rep)):
            if val:
                where.append(f"{col} = %s"); args.append(val)
        if q:
            where.append("(h.display_name ILIKE %s OR h.hcp_id ILIKE %s)"); args += [f"%{q}%", f"%{q}%"]
        order = "s.rank ASC" if sort == "uplift" else "(f.features_json->>'months_since_last_visit')::float DESC, s.rank ASC"
        w = " AND ".join(where)
        total = c.execute(f"SELECT count(*) AS n FROM hcp_score s JOIN hcp h USING (hcp_id) WHERE {w}", args).fetchone()["n"]
        rows = c.execute(f"""
            SELECT s.rank, h.hcp_id, h.display_name, h.specialty, h.territory_id, t.name AS territory_name, h.region,
                   h.rep_id, t.rep_name, s.uplift,
                   CASE WHEN s.rank <= %s THEN 'High' WHEN s.rank <= %s THEN 'Medium' ELSE 'Low' END AS band,
                   r.reason_text, r.fallback_used,
                   (f.features_json->>'visits_last_12m')::int AS visits_last_12m,
                   (f.features_json->>'months_since_last_visit')::float AS months_since_last_visit,
                   (SELECT action FROM rep_feedback fb WHERE fb.hcp_id=h.hcp_id AND fb.week=s.week
                     ORDER BY created_at DESC LIMIT 1) AS last_feedback
            FROM hcp_score s JOIN hcp h USING (hcp_id) JOIN territory t ON t.territory_id=h.territory_id
            JOIN hcp_feature f ON f.hcp_id=s.hcp_id AND f.week=s.week
            LEFT JOIN hcp_reason r ON r.hcp_id=s.hcp_id AND r.week=s.week
            WHERE {w} ORDER BY {order} LIMIT %s OFFSET %s""",
            [max(1, int(n * 0.03)), flag_cut] + args + [limit, offset]).fetchall()
        for r in rows:
            r["call_status"] = "Usual call" if (r["visits_last_12m"] or 0) >= 3 else "New"
        return {"week": week, "total": total, "flagged_cutoff": flag_cut, "items": rows}


@app.get("/hcps/{hcp_id}")
def hcp_detail(hcp_id: str, week: str | None = None):
    with conn() as c:
        week = _latest_week(c, week)
        h = c.execute("""SELECT h.*, t.name AS territory_name, t.rep_name FROM hcp h
                         JOIN territory t USING (territory_id) WHERE hcp_id=%s""", (hcp_id,)).fetchone()
        if not h:
            raise HTTPException(404, "HCP not found")
        run_id = _run_id(c, week)
        score = c.execute("SELECT rank, uplift, top_drivers_json FROM hcp_score WHERE run_id=%s AND hcp_id=%s",
                          (run_id, hcp_id)).fetchone()
        feat = c.execute("SELECT features_json FROM hcp_feature WHERE hcp_id=%s AND week=%s", (hcp_id, week)).fetchone()
        reason = c.execute("""SELECT reason_text, claims_json, verified, first_attempt_verified, fallback_used, attempts,
                              llm_model, prompt_version, tokens_in, tokens_out, created_at
                              FROM hcp_reason WHERE hcp_id=%s AND week=%s""", (hcp_id, week)).fetchone()
        rx = c.execute("SELECT period, period_label, trx, nbrx FROM rx_history WHERE hcp_id=%s ORDER BY period", (hcp_id,)).fetchall()
        calls = c.execute("""SELECT call_date, call_type, samples FROM call_activity WHERE hcp_id=%s
                             ORDER BY call_date DESC LIMIT 12""", (hcp_id,)).fetchall()
        fb = c.execute("""SELECT action, reason_code, rep_id, created_at FROM rep_feedback WHERE hcp_id=%s AND week=%s
                          ORDER BY created_at DESC LIMIT 5""", (hcp_id, week)).fetchall()
        drivers = score["top_drivers_json"] if score else []
        for d in drivers:
            d["label"] = FEATURE_LABELS.get(d["feature"], d["feature"])
        return {"week": week, "hcp": h, "rank": score["rank"] if score else None,
                "uplift": score["uplift"] if score else None, "drivers": drivers,
                "features": feat["features_json"] if feat else {}, "feature_labels": FEATURE_LABELS,
                "reason": reason, "rx_history": rx, "recent_calls": calls, "feedback": fb}


@app.get("/territories")
def territories():
    with conn() as c:
        rows = c.execute("""SELECT t.territory_id, t.name, t.region, t.rep_id, t.rep_name, count(h.hcp_id) AS hcp_count
                            FROM territory t LEFT JOIN hcp h USING (territory_id)
                            GROUP BY t.territory_id ORDER BY t.region, t.name""").fetchall()
        specs = [r["specialty"] for r in c.execute("SELECT DISTINCT specialty FROM hcp ORDER BY 1").fetchall()]
    return {"territories": rows, "regions": sorted({r["region"] for r in rows}), "specialties": specs}


@app.get("/metrics")
def metrics(week: str | None = None):
    with conn() as c:
        week = _latest_week(c, week)
        if not week:
            return {"week": None}
        runs = {r["model_type"]: r for r in c.execute(
            "SELECT DISTINCT ON (model_type) model_type, run_id, version, trained_at, metrics_json FROM model_run "
            "WHERE week=%s ORDER BY model_type, run_id DESC", (week,)).fetchall()}
        ver = c.execute("""SELECT count(*) AS n,
                              count(*) FILTER (WHERE first_attempt_verified) AS first_ok,
                              count(*) FILTER (WHERE verified) AS verified,
                              count(*) FILTER (WHERE fallback_used) AS fallback,
                              coalesce(sum(tokens_in),0) AS tokens_in, coalesce(sum(tokens_out),0) AS tokens_out,
                              count(*) FILTER (WHERE attempts > 0) AS llm_called
                           FROM hcp_reason WHERE week=%s""", (week,)).fetchone()
        flagged = runs["uplift"]["metrics_json"]["scoring"]["flagged_n"] if "uplift" in runs else 0
        adoption = c.execute("""
            SELECT
              (SELECT count(*) FROM usage_event WHERE week=%(w)s AND event_type='list_open') AS list_opens,
              (SELECT count(*) FROM usage_event WHERE week=%(w)s AND event_type='detail_view') AS detail_views,
              (SELECT count(DISTINCT hcp_id) FROM usage_event WHERE week=%(w)s AND event_type='contact_from_list') AS contacted,
              (SELECT count(DISTINCT rep_id) FROM usage_event WHERE week=%(w)s) AS active_reps,
              (SELECT count(*) FROM rep_feedback WHERE week=%(w)s AND action='will_call') AS will_call,
              (SELECT count(*) FROM rep_feedback WHERE week=%(w)s AND action='not_relevant') AS not_relevant
        """, {"w": week}).fetchone()
        reason_codes = c.execute("""SELECT reason_code, count(*) AS n FROM rep_feedback WHERE week=%s AND reason_code IS NOT NULL
                                    GROUP BY reason_code ORDER BY n DESC""", (week,)).fetchall()
        novelty = c.execute("""
            SELECT count(*) FILTER (WHERE (f.features_json->>'visits_last_12m')::int < 3) AS new_n, count(*) AS n
            FROM (SELECT DISTINCT hcp_id FROM usage_event WHERE week=%s AND event_type='contact_from_list') e
            JOIN hcp_feature f ON f.hcp_id=e.hcp_id AND f.week=%s""", (week, week)).fetchone()
        total_reps = c.execute("SELECT count(*) AS n FROM territory").fetchone()["n"]
        job = c.execute("SELECT job_type, status, detail, started_at, finished_at FROM job_status ORDER BY job_id DESC LIMIT 1").fetchone()
    return {
        "week": week,
        "current_model": runs.get("current", {}).get("metrics_json"),
        "uplift_model": runs.get("uplift", {}).get("metrics_json"),
        "runs": {k: {"run_id": v["run_id"], "version": v["version"], "trained_at": v["trained_at"]} for k, v in runs.items()},
        "verification": {**ver,
                         "first_attempt_pass_rate": round(ver["first_ok"] / ver["llm_called"] * 100, 1) if ver["llm_called"] else None,
                         "final_pass_rate": round(ver["verified"] / ver["n"] * 100, 1) if ver["n"] else None},
        "adoption": {**adoption, "flagged": flagged, "total_reps": total_reps,
                     "contact_rate_pct": round(adoption["contacted"] / flagged * 100, 2) if flagged else 0,
                     "novelty_pct": round(novelty["new_n"] / novelty["n"] * 100, 1) if novelty["n"] else None,
                     "reason_codes": reason_codes},
        "last_job": job,
    }


class Feedback(BaseModel):
    hcp_id: str = Field(max_length=16)
    rep_id: str = Field(max_length=16)
    action: Literal["will_call", "not_relevant", "contacted"]
    reason_code: str | None = Field(default=None, max_length=64)


class Event(BaseModel):
    rep_id: str = Field(max_length=16)
    event_type: Literal["list_open", "detail_view", "feedback", "contact_from_list", "filter_change"]
    hcp_id: str | None = Field(default=None, max_length=16)


@app.post("/feedback")
def feedback(fb: Feedback):
    week = current_week()
    with conn() as c:
        if not c.execute("SELECT 1 FROM hcp WHERE hcp_id=%s", (fb.hcp_id,)).fetchone():
            raise HTTPException(404, "HCP not found")
        week = _latest_week(c, None) or week
        r = c.execute("""INSERT INTO rep_feedback (hcp_id, rep_id, week, action, reason_code)
                         VALUES (%s,%s,%s,%s,%s) RETURNING feedback_id, created_at""",
                      (fb.hcp_id, fb.rep_id, week, fb.action, fb.reason_code)).fetchone()
        etype = "contact_from_list" if fb.action == "contacted" else "feedback"
        c.execute("INSERT INTO usage_event (rep_id, event_type, hcp_id, week) VALUES (%s,%s,%s,%s)",
                  (fb.rep_id, etype, fb.hcp_id, week))
        c.commit()
    return r


@app.post("/events")
def events(ev: Event):
    with conn() as c:
        week = _latest_week(c, None) or current_week()
        r = c.execute("INSERT INTO usage_event (rep_id, event_type, hcp_id, week) VALUES (%s,%s,%s,%s) RETURNING event_id",
                      (ev.rep_id, ev.event_type, ev.hcp_id, week)).fetchone()
        c.commit()
    return r


_job_lock = threading.Lock()


@app.post("/admin/rescore")
def rescore(step: Literal["score", "reasons", "all"] = "score", x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "job already running")

    def _work():
        from .pipeline import run
        try:
            run(step, job_type=f"rescore:{step}")
        finally:
            _job_lock.release()

    threading.Thread(target=_work, daemon=True).start()
    return {"started": True, "step": step}


@app.get("/admin/status")
def job_status():
    with conn() as c:
        return c.execute("SELECT * FROM job_status ORDER BY job_id DESC LIMIT 5").fetchall()
