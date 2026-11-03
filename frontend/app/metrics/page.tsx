"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Curve = { x: number[]; y: number[]; random: number[]; coefficient: number };
type CI = { estimate: number | null; ci_low: number | null; ci_high: number | null; n: number };
type M = {
  week: string | null;
  current_model: null | {
    validation_universe: { definition: string; n: number; auc: number; top_decile_precision: number };
    full_universe: { n: number; auc: number; top_decile_precision: number };
    rarely_visited_universe: { n: number; auc: number | null };
    feature_importance: { feature: string; importance: number }[];
    top_decile_heavy_visit_overlap_pct: number; base_rate_heavy_visit_pct: number;
  };
  uplift_model: null | {
    learner: string; train_n: number; holdout_n: number; holdout_randomized_n: number; propensity_auc: number;
    qini: { uplift: Curve; current: Curve; random: Curve };
    spearman_true_uplift: { uplift_model: number; current_model: number };
    top_decile_incremental_trx_randomized: { uplift_model: CI; current_model: CI };
    scoring: { flagged_n: number; current_top_decile_usual_pct: number; uplift_top_decile_usual_pct: number; top_decile_overlap_current_vs_uplift_pct: number; universe_usual_pct: number };
  };
  verification: { n: number; llm_called: number; first_attempt_pass_rate: number | null; final_pass_rate: number | null; fallback: number; tokens_in: number; tokens_out: number };
  adoption: { list_opens: number; detail_views: number; contacted: number; active_reps: number; total_reps: number; will_call: number; not_relevant: number; flagged: number; contact_rate_pct: number; novelty_pct: number | null; reason_codes: { reason_code: string; n: number }[] };
  last_job: null | { job_type: string; status: string; detail: string; started_at: string; finished_at: string | null };
};

function Qini({ q }: { q: M["uplift_model"] extends null ? never : NonNullable<M["uplift_model"]>["qini"] }) {
  const W = 640, H = 300, P = 40;
  const all = [...q.uplift.y, ...q.current.y, ...q.uplift.random, 0];
  const lo = Math.min(...all), hi = Math.max(...all, 1);
  const sx = (x: number) => P + x * (W - P * 1.5);
  const sy = (y: number) => H - P - ((y - lo) / (hi - lo || 1)) * (H - P * 1.6);
  const line = (c: Curve, key: "y" | "random" = "y") => c.x.map((x, i) => `${sx(x)},${sy(c[key][i])}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Qini curves">
      <line x1={P} y1={sy(0)} x2={W - P / 2} y2={sy(0)} stroke="hsla(220 40% 70% / .2)" />
      <line x1={P} y1={P / 2} x2={P} y2={H - P} stroke="hsla(220 40% 70% / .2)" />
      <polyline points={line(q.uplift, "random")} fill="none" stroke="hsl(220 15% 55%)" strokeDasharray="5 5" strokeWidth="1.5" />
      <polyline points={line(q.current)} fill="none" stroke="hsl(38 95% 60%)" strokeWidth="2.5" />
      <polyline points={line(q.random)} fill="none" stroke="hsl(220 15% 45%)" strokeWidth="1.2" opacity=".6" />
      <polyline points={line(q.uplift)} fill="none" stroke="hsl(168 80% 52%)" strokeWidth="3" />
      <text x={W / 2} y={H - 8} fill="hsl(220 15% 65%)" fontSize="12" textAnchor="middle">Share of randomized holdout targeted (ranked by model)</text>
      <text x={12} y={H / 2} fill="hsl(220 15% 65%)" fontSize="12" transform={`rotate(-90 12 ${H / 2})`} textAnchor="middle">Cumulative incremental TRx</text>
    </svg>
  );
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v}%`);

export default function Metrics() {
  const [m, setM] = useState<M | null>(null);
  const [err, setErr] = useState("");
  const [token, setToken] = useState("");
  const [msg, setMsg] = useState("");
  const load = () => api<M>("/metrics").then(d => { setM(d); setErr(""); }).catch(e => setErr(String(e)));
  useEffect(() => { load(); const id = setInterval(load, 10000); return () => clearInterval(id); }, []);

  const rescore = async (step: string) => {
    try {
      await api(`/admin/rescore?step=${step}`, { method: "POST", headers: { "X-Admin-Token": token } });
      setMsg(`Started ${step} job — metrics refresh automatically.`);
    } catch (e) { setMsg(String(e)); }
  };

  if (err) return <div className="card">Cannot reach API: {err}</div>;
  if (!m) return <div className="card"><div className="skeleton" /></div>;
  if (!m.week || !m.current_model || !m.uplift_model) return <div className="card empty">No model run yet. Run <span className="mono">python -m app.pipeline all</span>.</div>;
  const c = m.current_model, u = m.uplift_model, s = u.scoring, a = m.adoption, v = m.verification;
  const ti = u.top_decile_incremental_trx_randomized;
  const ciTxt = (x: CI) => x.estimate === null ? "—" : `${x.estimate} [${x.ci_low}, ${x.ci_high}]`;

  return (
    <>
      <section className="hero">
        <div>
          <h1>Diagnosis & <span className="grad">proof the fix works</span></h1>
          <p>Every number below is computed from the Postgres tables by the latest training run (week {m.week}). Ground truth is used only for the rank-correlation check.</p>
        </div>
        <div className="row small muted"><span className="live" aria-hidden /> Auto-refresh 10s · last job: {m.last_job?.job_type} {m.last_job?.status}</div>
      </section>

      <div className="kpis">
        <div className="card kpi"><div className="label">Qini · uplift model</div><div className="value" style={{ color: "var(--accent)" }}>{u.qini.uplift.coefficient}</div><div className="sub">current {u.qini.current.coefficient} · random {u.qini.random.coefficient}</div></div>
        <div className="card kpi"><div className="label">Spearman vs planted uplift</div><div className="value">{u.spearman_true_uplift.uplift_model}</div><div className="sub">current model {u.spearman_true_uplift.current_model} · target &gt; 0.5</div></div>
        <div className="card kpi"><div className="label">Reason verification</div><div className="value">{pct(v.first_attempt_pass_rate)}</div><div className="sub">first-attempt LLM pass · {pct(v.final_pass_rate)} after fallback · {v.fallback} fallbacks</div></div>
        <div className="card kpi"><div className="label">Contact rate (flagged)</div><div className="value">{a.contact_rate_pct}%</div><div className="sub">{a.contacted} of {a.flagged} · baseline ≈4% · target 25%</div></div>
      </div>

      <div className="grid2">
        <div className="card">
          <h3 className="section" style={{ marginTop: 0 }}>Qini curve (held-out randomized pilot, n={u.holdout_randomized_n})</h3>
          <Qini q={u.qini} />
          <div className="row small" style={{ marginTop: 8 }}>
            <span className="badge b-high">Uplift model</span><span className="badge b-new">Current model</span><span className="badge b-usual">Random</span>
          </div>
        </div>
        <div className="card">
          <h3 style={{ fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: ".08em", fontFamily: "Inter" }}>Current model: two universes</h3>
          <table style={{ marginTop: 10 }}>
            <thead><tr><th>Universe</th><th>n</th><th>AUC</th><th>Top-decile precision</th></tr></thead>
            <tbody>
              <tr style={{ cursor: "default" }}><td>Validation ({c.validation_universe.definition})</td><td>{c.validation_universe.n}</td><td>{c.validation_universe.auc}</td><td>{(c.validation_universe.top_decile_precision * 100).toFixed(1)}%</td></tr>
              <tr style={{ cursor: "default" }}><td>Full universe</td><td>{c.full_universe.n}</td><td>{c.full_universe.auc}</td><td>{(c.full_universe.top_decile_precision * 100).toFixed(1)}%</td></tr>
              <tr style={{ cursor: "default" }}><td>Rarely visited (&lt;3 visits)</td><td>{c.rarely_visited_universe.n}</td><td>{c.rarely_visited_universe.auc ?? "—"}</td><td>—</td></tr>
            </tbody>
          </table>
          <h3 style={{ fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: ".08em", fontFamily: "Inter", marginTop: 18 }}>Overlap diagnostics</h3>
          <ul className="clean small">
            <li>Current top decile that is heavily visited: <b>{c.top_decile_heavy_visit_overlap_pct}%</b> (base rate {c.base_rate_heavy_visit_pct}%)</li>
            <li>Current top decile already &quot;usual calls&quot;: <b>{s.current_top_decile_usual_pct}%</b> vs uplift top decile <b>{s.uplift_top_decile_usual_pct}%</b> (universe {s.universe_usual_pct}%)</li>
            <li>Top-decile overlap, current vs uplift list: <b>{s.top_decile_overlap_current_vs_uplift_pct}%</b></li>
            <li>Top-decile incremental TRx/quarter (randomized, 95% CI): uplift <b>{ciTxt(ti.uplift_model)}</b>, current <b>{ciTxt(ti.current_model)}</b></li>
            <li>Propensity model AUC (selection bias strength): <b>{u.propensity_auc}</b></li>
          </ul>
        </div>
      </div>

      <div className="grid2 section">
        <div className="card">
          <h3 style={{ fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: ".08em", fontFamily: "Inter" }}>Current model feature importance</h3>
          {c.feature_importance.map(f => (
            <div className="driver" key={f.feature}><span className="mono">{f.feature}</span><b>{(f.importance * 100).toFixed(1)}%</b>
              <div className="bar"><span style={{ width: `${f.importance / c.feature_importance[0].importance * 100}%` }} /></div></div>
          ))}
        </div>
        <div className="card">
          <h3 style={{ fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: ".08em", fontFamily: "Inter" }}>Adoption (live, week {m.week})</h3>
          <div className="kpis" style={{ marginTop: 10 }}>
            <div className="kpi"><div className="label">List opens</div><div className="value">{a.list_opens}</div></div>
            <div className="kpi"><div className="label">Detail views</div><div className="value">{a.detail_views}</div></div>
            <div className="kpi"><div className="label">Active reps</div><div className="value">{a.active_reps}/{a.total_reps}</div></div>
            <div className="kpi"><div className="label">Will call / not rel.</div><div className="value">{a.will_call}/{a.not_relevant}</div></div>
            <div className="kpi"><div className="label">Novelty</div><div className="value">{pct(a.novelty_pct)}</div></div>
          </div>
          {a.reason_codes.length > 0 && <ul className="clean small">{a.reason_codes.map(r => <li key={r.reason_code}>{r.reason_code}: {r.n}</li>)}</ul>}
          <div className="small muted">LLM tokens this week: {v.tokens_in} in / {v.tokens_out} out · {v.llm_called} HCPs sent to Mistral</div>
        </div>
      </div>

      <div className="grid2 section">
        <div className="card">
          <h3 style={{ fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: ".08em", fontFamily: "Inter" }}>Test design: new list vs current list</h3>
          <ul className="clean small">
            <li><b>Unit:</b> territory (fallback: within-rep 10% holdout)</li>
            <li><b>Assignment:</b> matched territory pairs by size, baseline Rx, specialty mix</li>
            <li><b>Duration:</b> 6–8 weeks plus pre-period baseline</li>
            <li><b>Primary:</b> contact rate on flagged HCPs; incremental Rx per call</li>
            <li><b>Analysis:</b> difference-in-differences on matched pairs with CIs</li>
            <li><b>Guardrails:</b> total call volume, rep satisfaction, key-account coverage</li>
          </ul>
        </div>
        <div className="card">
          <h3 style={{ fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: ".08em", fontFamily: "Inter" }}>Admin: weekly batch</h3>
          <p className="small muted">Re-trains, rescores and regenerates reasons. Requires the ADMIN_TOKEN from your .env (never stored in the browser).</p>
          <div className="row">
            <input id="admin-token" type="password" placeholder="Admin token" value={token} onChange={e => setToken(e.target.value)} />
            <button id="btn-rescore" onClick={() => rescore("score")} disabled={!token}>Rescore + reasons</button>
            <button id="btn-reasons" className="primary" onClick={() => rescore("reasons")} disabled={!token}>Reasons only</button>
          </div>
          {msg && <div className="small" style={{ marginTop: 8 }}>{msg}</div>}
          {m.last_job && <div className="small muted mono" style={{ marginTop: 8, wordBreak: "break-word" }}>{m.last_job.status}: {m.last_job.detail?.slice(0, 300)}</div>}
        </div>
      </div>
    </>
  );
}
