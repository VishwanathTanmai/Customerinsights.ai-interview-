"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api, Territory } from "@/lib/api";
import { useRep } from "@/lib/rep";

export default function Nav() {
  const path = usePathname();
  const [rep, setRep] = useRep();
  const [terrs, setTerrs] = useState<Territory[]>([]);
  useEffect(() => { api<{ territories: Territory[] }>("/territories").then(d => setTerrs(d.territories)).catch(() => { }); }, []);
  return (
    <nav className="nav" aria-label="Main">
      <div className="brand"><div className="logo" aria-hidden /> Uplift Planner</div>
      <div className="navlinks">
        <Link href="/" className={path === "/" ? "active" : ""} id="nav-list">Call list</Link>
        <Link href="/metrics" className={path === "/metrics" ? "active" : ""} id="nav-metrics">Metrics & diagnosis</Link>
      </div>
      <div className="spacer" />
      <label className="row small muted" htmlFor="rep-select">
        Acting as
        <select id="rep-select" value={rep} onChange={e => setRep(e.target.value)}>
          <option value="">Select rep…</option>
          {terrs.map(t => <option key={t.rep_id} value={t.rep_id}>{t.rep_name} · {t.name}</option>)}
        </select>
      </label>
    </nav>
  );
}
