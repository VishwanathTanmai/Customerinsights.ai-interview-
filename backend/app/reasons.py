"""FR4: Mistral-generated reasons with programmatic claim verification and template fallback."""
from __future__ import annotations

import asyncio
import json
import re

import httpx

from .config import (MISTRAL_API_KEY, MISTRAL_MODEL, MISTRAL_TEMPERATURE, MISTRAL_URL, PROMPT_VERSION, REASONS_CONCURRENCY,
                     REASONS_MAX, TOKEN_BUDGET, current_week)
from .db import connect
from .features import FEATURE_LABELS

BANNED = re.compile(r"\b(uplift|guarantee\w*|promise\w*|will (increase|grow|prescribe)|patient(s)?\b(?! volume)|"
                    r"efficacy|effective|cure\w*|off-label|safe(ty)?|clinical)\b", re.I)
MAX_WORDS = 40

SYSTEM_PROMPT = (
    "You write internal call-planning notes for pharmaceutical field reps. "
    "Rules: use ONLY the facts provided; do not introduce any number that is not in the facts; "
    "no clinical, efficacy, safety or off-label statements; no patient information; never promise outcomes "
    "or mention scores; write one or two plain sentences, at most 40 words. "
    "Return JSON with 'reason' (string) and 'claims' (array of {field, value}) listing every fact you used, "
    "with field names exactly as given."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"field": {"type": "string"}, "value": {"type": ["number", "string"]}},
            "required": ["field", "value"], "additionalProperties": False}},
    },
    "required": ["reason", "claims"], "additionalProperties": False,
}


def build_facts(specialty: str, features: dict, drivers: list[dict]) -> dict:
    facts = {"specialty": specialty}
    for d in drivers:
        f = d["feature"]
        if f in ("is_derm", "is_np_pa"):
            continue
        facts[f] = features[f]
    return facts


def _num_ok(claimed, actual) -> bool:
    try:
        a, b = float(claimed), float(actual)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(0.5, abs(b) * 0.02)


def verify(text: str, claims: list[dict], facts: dict, stored: dict) -> tuple[bool, str | None]:
    """Every claim must reference a supplied field whose value matches the stored row within tolerance."""
    if not text or not claims:
        return False, "empty reason or claims"
    if len(text.split()) > MAX_WORDS:
        return False, "too long"
    if BANNED.search(text):
        return False, f"banned term: {BANNED.search(text).group(0)}"
    for cl in claims:
        f, v = cl.get("field"), cl.get("value")
        if f not in facts:
            return False, f"unknown field {f}"
        actual = stored.get(f, facts.get(f)) if f != "specialty" else facts["specialty"]
        if isinstance(actual, str):
            if str(v).strip().lower() != actual.lower():
                return False, f"mismatch {f}"
        elif not _num_ok(v, actual):
            return False, f"mismatch {f}: {v} vs {actual}"
    # every number written in the text must correspond to a supplied fact or standard feature window (1, 2, 3, 6, 9, 12 months)
    time_windows = [1.0, 2.0, 3.0, 6.0, 9.0, 12.0]
    allowed = [float(x) for x in facts.values() if isinstance(x, (int, float))] + time_windows
    for tok in re.findall(r"-?\d+(?:\.\d+)?", text):
        if not any(_num_ok(tok, a) for a in allowed):
            return False, f"unsupported number {tok}"
    return True, None


def template_reason(facts: dict) -> tuple[str, list[dict]]:
    parts, claims = [], []
    for f, v in facts.items():
        if f == "specialty":
            continue
        parts.append(f"{FEATURE_LABELS.get(f, f).lower()} is {v}")
        claims.append({"field": f, "value": v})
        if len(parts) == 2:
            break
    claims.append({"field": "specialty", "value": facts["specialty"]})
    text = f"{facts['specialty']} prescriber: " + "; ".join(parts) + "." if parts else f"{facts['specialty']} prescriber."
    return text[0].upper() + text[1:], claims


async def call_mistral(client: httpx.AsyncClient, facts: dict, retries: int = 3) -> dict:
    labelled = {k: {"value": v, "meaning": FEATURE_LABELS.get(k, k)} for k, v in facts.items()}
    body = {
        "model": MISTRAL_MODEL, "temperature": MISTRAL_TEMPERATURE, "max_tokens": 200,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": "Facts:\n" + json.dumps(labelled)}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "reason", "schema": SCHEMA, "strict": True}},
    }
    delay = 1.5
    for attempt in range(retries):
        try:
            r = await client.post(MISTRAL_URL, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            data = r.json()
            parsed = json.loads(data["choices"][0]["message"]["content"])
            u = data.get("usage", {})
            return {"reason": parsed.get("reason", ""), "claims": parsed.get("claims", []),
                    "tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0)}
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


async def _process(sem, client, row, week, run_id, budget):
    facts = build_facts(row["specialty"], row["features"], row["drivers"])
    rec = {"tokens_in": 0, "tokens_out": 0, "attempts": 0, "error": None, "first_ok": False}
    text, claims, ok = None, [], False
    if client is not None:
        async with sem:
            for attempt in range(2):  # generate, then regenerate once
                if budget["used"] >= TOKEN_BUDGET:
                    rec["error"] = "token budget exhausted"
                    break
                try:
                    res = await call_mistral(client, facts)
                except Exception as e:  # noqa: BLE001 - never block the list
                    rec["error"] = f"mistral error: {type(e).__name__}"
                    break
                rec["attempts"] += 1
                rec["tokens_in"] += res["tokens_in"]
                rec["tokens_out"] += res["tokens_out"]
                budget["used"] += res["tokens_in"] + res["tokens_out"]
                ok, why = verify(res["reason"], res["claims"], facts, row["features"])
                if ok:
                    text, claims = res["reason"].strip(), res["claims"]
                    rec["first_ok"] = attempt == 0
                    break
                rec["error"] = f"verification failed: {why}"
    else:
        rec["error"] = "MISTRAL_API_KEY not set"
    fallback = not ok
    if fallback:
        text, claims = template_reason(facts)
    return (row["hcp_id"], week, run_id, text, json.dumps(claims), True, rec["first_ok"], fallback,
            rec["attempts"], MISTRAL_MODEL if rec["attempts"] else "template", PROMPT_VERSION,
            rec["tokens_in"], rec["tokens_out"], rec["error"])


async def _run(rows, week, run_id):
    sem = asyncio.Semaphore(REASONS_CONCURRENCY)
    budget = {"used": 0}
    if MISTRAL_API_KEY:
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "HTTP-Referer": "http://localhost:3000", "X-Title": "Uplift Prototype"}
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            return await asyncio.gather(*[_process(sem, client, r, week, run_id, budget) for r in rows]), budget
    return await asyncio.gather(*[_process(sem, None, r, week, run_id, budget) for r in rows]), budget


def generate_reasons(force: bool = False) -> dict:
    week = current_week()
    with connect() as c:
        run = c.execute("SELECT run_id FROM model_run WHERE model_type='uplift' AND week=%s ORDER BY run_id DESC LIMIT 1",
                        (week,)).fetchone()
        if not run:
            raise RuntimeError("No uplift run for this week; run scoring first")
        run_id = run["run_id"]
        cache_clause = "" if force else \
            "AND NOT EXISTS (SELECT 1 FROM hcp_reason r WHERE r.hcp_id=s.hcp_id AND r.week=s.week AND r.run_id=s.run_id AND r.fallback_used=false)"
        rows = c.execute(f"""
            SELECT s.hcp_id, h.specialty, f.features_json AS features, s.top_drivers_json AS drivers
            FROM hcp_score s JOIN hcp h USING (hcp_id)
            JOIN hcp_feature f ON f.hcp_id=s.hcp_id AND f.week=s.week
            WHERE s.run_id=%s AND s.rank <= %s {cache_clause}
            ORDER BY s.rank""", (run_id, REASONS_MAX)).fetchall()

    results, budget = asyncio.run(_run(rows, week, run_id)) if rows else ([], {"used": 0})

    with connect() as c:
        for r in results:
            c.execute("""
                INSERT INTO hcp_reason (hcp_id, week, run_id, reason_text, claims_json, verified, first_attempt_verified,
                    fallback_used, attempts, llm_model, prompt_version, tokens_in, tokens_out, error)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (hcp_id, week) DO UPDATE SET run_id=EXCLUDED.run_id, reason_text=EXCLUDED.reason_text,
                    claims_json=EXCLUDED.claims_json, verified=EXCLUDED.verified,
                    first_attempt_verified=EXCLUDED.first_attempt_verified, fallback_used=EXCLUDED.fallback_used,
                    attempts=EXCLUDED.attempts, llm_model=EXCLUDED.llm_model, prompt_version=EXCLUDED.prompt_version,
                    tokens_in=EXCLUDED.tokens_in, tokens_out=EXCLUDED.tokens_out, error=EXCLUDED.error, created_at=now()
            """, r)
        c.commit()
    return {"processed": len(results), "fallbacks": sum(1 for r in results if r[7]), "tokens": budget["used"],
            "mistral_enabled": bool(MISTRAL_API_KEY)}


if __name__ == "__main__":
    print(generate_reasons())
