"""FR1: Seeded synthetic HCP universe generator.

Writes territories, HCPs, monthly Rx (15 months), calls, emails, the randomized
pilot membership, and the hidden true_uplift (restricted schema) to Postgres.

Timeline (month index):
  1-9   pre-period (features for training)
  10-12 treatment window (visited or not)
  13-15 outcome window (Rx response incl. incremental visit effect)
Scoring for the current week uses the latest 9-month window (7-15).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .config import SEED
from .db import connect, init_schema

N_HCP = 4000
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
TERR_PER_REGION = 8
SPECIALTIES = ["Dermatology", "Primary Care", "Derm NP/PA"]
SPEC_P = [0.40, 0.45, 0.15]
FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Jamie", "Drew",
         "Rowan", "Sage", "Reese", "Parker", "Hayden", "Emerson", "Finley", "Kendall", "Logan", "Skyler"]
LAST = ["Arden", "Bellamy", "Calder", "Dorsey", "Ellery", "Fairfax", "Garland", "Hollis", "Ingram",
        "Jessup", "Keaton", "Lindqvist", "Marlowe", "Nash", "Orwell", "Prescott", "Quimby", "Rainier",
        "Sutton", "Thorne", "Underhill", "Vance", "Whitlock", "Yardley", "Zeller"]
CITIES = {
    "Northeast": ["Boston", "Hartford", "Albany", "Providence", "Newark", "Philadelphia", "Pittsburgh", "Portland ME"],
    "Southeast": ["Atlanta", "Miami", "Charlotte", "Nashville", "Tampa", "Raleigh", "Orlando", "Birmingham"],
    "Midwest": ["Chicago", "Detroit", "Columbus", "Indianapolis", "Milwaukee", "Minneapolis", "St. Louis", "Omaha"],
    "Southwest": ["Dallas", "Houston", "Austin", "Phoenix", "Tucson", "Albuquerque", "San Antonio", "Oklahoma City"],
    "West": ["Los Angeles", "San Diego", "San Francisco", "Seattle", "Portland OR", "Denver", "Salt Lake City", "Las Vegas"],
}


def _month_start(anchor: date, months_back: int) -> date:
    y, m = anchor.year, anchor.month - months_back
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, 1)


def generate(seed: int = SEED, effect_scale: float = 1.0, noise_scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    init_schema()

    today = date.today()
    # period 15 = last complete month
    month_dates = {p: _month_start(today, 16 - p) for p in range(1, 16)}

    # Territories and reps
    territories = []
    for r in REGIONS:
        for i, city in enumerate(CITIES[r][:TERR_PER_REGION]):
            tid = f"T{len(territories)+1:03d}"
            rep_id = f"R{len(territories)+1:03d}"
            rep_name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            territories.append((tid, city, r, rep_id, rep_name))
    t_idx = rng.integers(0, len(territories), N_HCP)

    spec = rng.choice(SPECIALTIES, N_HCP, p=SPEC_P)
    is_derm = spec == "Dermatology"
    is_np = spec == "Derm NP/PA"

    # Baseline monthly TRx: lognormal; derm >> primary care (CMS Part D-shaped long tail)
    mu = np.where(is_derm, np.log(9.0), np.where(is_np, np.log(5.0), np.log(1.6)))
    base = rng.lognormal(mu, 0.75)
    trend = rng.normal(0.02, 0.18, N_HCP)  # annualized growth
    panel = np.clip((base * rng.uniform(25, 45, N_HCP) + rng.normal(300, 120, N_HCP)), 60, None).astype(int)
    email_affinity = rng.beta(2, 5, N_HCP)

    # Hidden true uplift (monthly TRx gained from being visited in treatment window)
    pct = base.argsort().argsort() / (N_HCP - 1)
    mid_bump = np.exp(-((pct - 0.55) ** 2) / (2 * 0.17 ** 2))
    top_damp = np.where(pct > 0.9, np.clip((1.0 - pct) / 0.1, 0, 1) * 0.15, 1.0)
    spec_mult = np.where(is_derm, 1.0, np.where(is_np, 0.6, 0.15))
    trend_mult = 1.0 / (1.0 + np.exp(-trend / 0.08))
    true_uplift = effect_scale * 3.2 * mid_bump * top_damp * spec_mult * (0.35 + 1.3 * trend_mult)
    true_uplift = np.clip(true_uplift + rng.normal(0, 0.15, N_HCP) * spec_mult, 0, None)

    # Randomized pilot: 18% of HCPs get visits assigned at random (p=0.5) in treatment window
    pilot = rng.random(N_HCP) < 0.18

    # Habitual visit intensity biased toward high prescribers (selection problem)
    habit_logit = -2.2 + 1.25 * np.log(base + 0.5) + 0.4 * is_derm
    habit_rate = 1 / (1 + np.exp(-habit_logit))  # prob of a visit per month

    hcps, rx_rows, call_rows, email_rows = [], [], [], []
    for i in range(N_HCP):
        tid, city, region, rep_id, _ = territories[t_idx[i]]
        hid = f"H{i+1:05d}"
        name = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
        hcps.append((hid, name, spec[i], tid, region, rep_id, int(panel[i])))

        visits_by_month = np.zeros(16, dtype=int)
        treated = False
        for p in range(1, 16):
            if 10 <= p <= 12 and pilot[i]:
                continue
            n = rng.binomial(2, habit_rate[i] * 0.6)
            visits_by_month[p] = n
        if pilot[i]:
            if rng.random() < 0.5:
                p = int(rng.integers(10, 13))
                visits_by_month[p] = max(1, visits_by_month[p]) + int(rng.random() < 0.4)
        treated = visits_by_month[10:13].sum() > 0

        for p in range(1, 16):
            for _ in range(visits_by_month[p]):
                d = month_dates[p] + timedelta(days=int(rng.integers(0, 28)))
                rnd = bool(pilot[i] and 10 <= p <= 12)
                ctype = rng.choice(["In-person", "Virtual", "Lunch & learn"], p=[0.7, 0.22, 0.08])
                samples = int(rng.poisson(4 if spec[i] != "Primary Care" else 1.5))
                call_rows.append((hid, rep_id, d, p, str(ctype), samples, rnd))

            lvl = base[i] * (1 + trend[i]) ** (p / 12.0)
            if p >= 13 and treated:
                lvl += true_uplift[i]
            trx = int(rng.poisson(max(lvl * rng.lognormal(0, 0.12 * noise_scale), 0.01)))
            nbrx = int(rng.binomial(trx, 0.18 + (0.1 if trend[i] > 0.1 else 0.0)))
            rx_rows.append((hid, p, month_dates[p].strftime("%Y-%m"), trx, nbrx))

            sent = int(rng.integers(1, 5))
            opened = int(rng.binomial(sent, email_affinity[i]))
            clicked = int(rng.binomial(opened, 0.25))
            email_rows.append((hid, p, sent, opened, clicked))

    with connect() as c:
        c.execute("""TRUNCATE usage_event, rep_feedback, hcp_reason, hcp_score, hcp_feature, model_run,
                     random_pilot, email_engagement, call_activity, rx_history, hcp, territory,
                     restricted.ground_truth RESTART IDENTITY CASCADE""")
        with c.cursor() as cur:
            with cur.copy("COPY territory (territory_id,name,region,rep_id,rep_name) FROM STDIN") as cp:
                for r in territories:
                    cp.write_row(r)
            with cur.copy("COPY hcp (hcp_id,display_name,specialty,territory_id,region,rep_id,panel_size_proxy) FROM STDIN") as cp:
                for r in hcps:
                    cp.write_row(r)
            with cur.copy("COPY rx_history (hcp_id,period,period_label,trx,nbrx) FROM STDIN") as cp:
                for r in rx_rows:
                    cp.write_row(r)
            with cur.copy("COPY call_activity (hcp_id,rep_id,call_date,period,call_type,samples,assigned_randomly) FROM STDIN") as cp:
                for r in call_rows:
                    cp.write_row(r)
            with cur.copy("COPY email_engagement (hcp_id,period,sent,opened,clicked) FROM STDIN") as cp:
                for r in email_rows:
                    cp.write_row(r)
            with cur.copy("COPY random_pilot (hcp_id) FROM STDIN") as cp:
                for i in np.where(pilot)[0]:
                    cp.write_row((f"H{i+1:05d}",))
            with cur.copy("COPY restricted.ground_truth (hcp_id,true_uplift) FROM STDIN") as cp:
                for i in range(N_HCP):
                    cp.write_row((f"H{i+1:05d}", float(true_uplift[i])))
        c.commit()
    return {"hcps": N_HCP, "calls": len(call_rows), "rx_rows": len(rx_rows), "pilot": int(pilot.sum())}


if __name__ == "__main__":
    print(generate())
