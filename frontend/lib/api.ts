const BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export function logEvent(rep_id: string, event_type: string, hcp_id?: string) {
  if (!rep_id) return;
  api("/events", { method: "POST", body: JSON.stringify({ rep_id, event_type, hcp_id }) }).catch(() => {});
}

export type HcpRow = {
  rank: number; hcp_id: string; display_name: string; specialty: string; territory_id: string;
  territory_name: string; region: string; rep_id: string; rep_name: string; uplift: number;
  band: "High" | "Medium" | "Low"; reason_text: string | null; fallback_used: boolean | null;
  visits_last_12m: number; months_since_last_visit: number; last_feedback: string | null; call_status: string;
};

export type Territory = { territory_id: string; name: string; region: string; rep_id: string; rep_name: string; hcp_count: number };
