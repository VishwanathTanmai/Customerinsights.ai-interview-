"use client";
import { useEffect, useState } from "react";
import { api, logEvent } from "@/lib/api";
import { useRep } from "@/lib/rep";

type Detail = {
  week: string; rank: number; uplift: number;
  hcp: { hcp_id: string; display_name: string; specialty: string; territory_name: string; rep_name: string; region: string; panel_size_proxy: number };
  drivers: { feature: string; label: string; contribution: number; value: number }[];
  features: Record<string, number>; feature_labels: Record<string, string>;
  reason: null | { reason_text: string; claims_json: { field: string; value: number | string }[]; verified: boolean; first_attempt_verified: boolean; fallback_used: boolean; llm_model: string; prompt_version: string; tokens_in: number; tokens_out: number };
  rx_history: { period: number; period_label: string; trx: number; nbrx: number }[];
  recent_calls: { call_date: string; call_type: string; samples: number }[];
  feedback: { action: string; reason_code: string | null; rep_id: string; created_at: string }[];
};

const CODES = {
  will_call: ["Fits route this week", "Reason is compelling", "Overdue follow-up"],
  not_relevant: ["Already call regularly", "Access restricted", "Not a fit for brand", "Reason inaccurate"],
};

function Spark({ data }: { data: Detail["rx_history"] }) {
  if (!data.length) return null;
  const W = 400, H = 90, max = Math.max(...data.map(d => d.trx), 1);
  const pts = data.map((d, i) => `${(i / (data.length - 1)) * W},${H - (d.trx / max) * (H - 10) - 5}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} width="100%" role="img" aria-label="Monthly TRx history">
      <defs><linearGradient id="sg" x1="0" x2="1"><stop offset="0" stopColor="hsl(168 80% 52%)" /><stop offset="1" stopColor="hsl(262 85% 68%)" /></linearGradient></defs>
      <polyline points={pts} fill="none" stroke="url(#sg)" strokeWidth="2.5" strokeLinejoin="round" />
      {data.map((d, i) => <circle key={i} cx={(i / (data.length - 1)) * W} cy={H - (d.trx / max) * (H - 10) - 5} r="2.5" fill="hsl(168 80% 52%)"><title>{d.period_label}: {d.trx} TRx</title></circle>)}
      <text x="0" y={H + 15} fill="hsl(220 15% 65%)" fontSize="11">{data[0].period_label}</text>
      <text x={W} y={H + 15} fill="hsl(220 15% 65%)" fontSize="11" textAnchor="end">{data[data.length - 1].period_label}</text>
    </svg>
  );
}

export default function HcpPanel({ hcpId, onClose, onChanged }: { hcpId: string; onClose: () => void; onChanged: () => void }) {
  const [rep] = useRep();
  const [d, setD] = useState<Detail | null>(null);
  const [action, setAction] = useState<"will_call" | "not_relevant" | "">("");
  const [code, setCode] = useState("");
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () => api<Detail>(`/hcps/${hcpId}`).then(setD);
  useEffect(() => { setD(null); setAction(""); setCode(""); load(); if (rep) logEvent(rep, "detail_view", hcpId); }, [hcpId]); // eslint-disable-line

  const submit = async (act: string, rc?: string) => {
    if (!rep) { setToast("Select who you are (top right) first"); return; }
    setBusy(true);
    try {
      await api("/feedback", { method: "POST", body: JSON.stringify({ hcp_id: hcpId, rep_id: rep, action: act, reason_code: rc || null }) });
      setToast("Saved"); setAction(""); setCode(""); load(); onChanged();
    } catch (e) { setToast(String(e)); } finally { setBusy(false); setTimeout(() => setToast(""), 2500); }
  };

  if (!d) return <aside className="card panel"><div className="skeleton" style={{ height: 24, width: "60%" }} /><div className="skeleton" style={{ marginTop: 14 }} /></aside>;
  const maxC = Math.max(...d.drivers.map(x => Math.abs(x.contribution)), 0.0001);
  const rx = d.rx_history;
  const last3 = rx.slice(-3).reduce((a, b) => a + b.trx, 0), prev3 = rx.slice(-6, -3).reduce((a, b) => a + b.trx, 0);

  return (
    <aside className="card panel" aria-label="HCP detail">
      <div className="row between">
        <div><h2>{d.hcp.display_name}</h2><div className="small muted">{d.hcp.hcp_id} · {d.hcp.specialty} · {d.hcp.territory_name}, {d.hcp.region}</div></div>
        <button id="close-panel" className="ghost" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <span className="badge b-high">Rank #{d.rank}</span>
        <span className="badge b-medium">+{d.uplift?.toFixed(2)} TRx/mo est.</span>
        <span className={`badge ${(d.features.visits_last_12m ?? 0) >= 3 ? "b-usual" : "b-new"}`}>{(d.features.visits_last_12m ?? 0) >= 3 ? "Usual call" : "New"}</span>
      </div>

      <div className="section">
        <h3>Why this HCP</h3>
        {d.reason ? (
          <>
            <div className="quote">{d.reason.reason_text}</div>
            <div className="row small muted" style={{ marginTop: 8 }}>
              <span className={`badge ${d.reason.fallback_used ? "b-fb" : "b-ok"}`}>{d.reason.fallback_used ? "Template fallback" : d.reason.first_attempt_verified ? "Verified · 1st try" : "Verified · regenerated"}</span>
              <span>{d.reason.llm_model} · prompt {d.reason.prompt_version} · {d.reason.tokens_in + d.reason.tokens_out} tokens</span>
            </div>
            <div className="small muted" style={{ marginTop: 6 }}>Claims checked: {d.reason.claims_json.map(c => `${c.field}=${c.value}`).join(", ")}</div>
          </>
        ) : <div className="muted small">Outside this week&apos;s reason batch.</div>}
      </div>

      <div className="section">
        <h3>Top drivers of uplift</h3>
        {d.drivers.map(x => (
          <div className="driver" key={x.feature}>
            <span>{x.label}</span><b>{x.value}</b>
            <div className="bar"><span style={{ width: `${Math.abs(x.contribution) / maxC * 100}%` }} /></div>
          </div>
        ))}
      </div>

      <div className="section">
        <h3>Baseline vs trend</h3>
        <div className="row small"><span>Last 3 mo: <b>{last3}</b> TRx</span><span>Prior 3 mo: <b>{prev3}</b></span><span>Trend: <b>{d.features.rx_trend_pct}%</b></span></div>
        <Spark data={rx} />
      </div>

      <div className="section">
        <h3>Recent call history</h3>
        {d.recent_calls.length ? (
          <table><tbody>{d.recent_calls.slice(0, 6).map((c, i) => <tr key={i} style={{ cursor: "default" }}><td>{c.call_date}</td><td>{c.call_type}</td><td className="muted">{c.samples} samples</td></tr>)}</tbody></table>
        ) : <div className="muted small">No recorded calls.</div>}
      </div>

      <div className="section">
        <h3>Your feedback</h3>
        <div className="row">
          <button id="fb-will-call" className={action === "will_call" ? "primary" : ""} onClick={() => setAction("will_call")} disabled={busy}>Will call</button>
          <button id="fb-not-relevant" className={action === "not_relevant" ? "primary" : ""} onClick={() => setAction("not_relevant")} disabled={busy}>Not relevant</button>
          <button id="fb-contacted" onClick={() => submit("contacted")} disabled={busy}>Mark contacted</button>
        </div>
        {action && (
          <div className="row" style={{ marginTop: 10 }}>
            <select id="fb-reason" aria-label="Reason code" value={code} onChange={e => setCode(e.target.value)}>
              <option value="">Reason code…</option>{CODES[action].map(c => <option key={c}>{c}</option>)}
            </select>
            <button id="fb-submit" className="primary" disabled={!code || busy} onClick={() => submit(action, code)}>Submit</button>
          </div>
        )}
        {d.feedback.length > 0 && <ul className="clean small muted" style={{ marginTop: 10 }}>{d.feedback.map((f, i) => <li key={i}>{f.action.replace("_", " ")}{f.reason_code ? ` — ${f.reason_code}` : ""} · {f.rep_id} · {new Date(f.created_at).toLocaleString()}</li>)}</ul>}
      </div>
      {toast && <div className="toast" role="status">{toast}</div>}
    </aside>
  );
}
