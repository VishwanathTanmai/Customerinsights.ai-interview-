# PRD: Uplift-Based HCP Targeting Prototype

**Case:** Case 4, Low Field Adoption **Domain:** Pharma, dermatology, HCP targeting **Author:** FDE Associate **Status:** Draft v1 **Stack:** Python (FastAPI), PostgreSQL, Mistral API, Next.js (React, TypeScript)

## 1. Summary

A specialty pharma client launched a topical dermatology therapy nine months ago. A targeting model scores about 4,000 HCPs weekly and feeds rep call-planning lists. It has been live for five months. Its validation metrics are strong (AUC 0.86, top-decile precision 71%), yet for three months running reps have contacted under 5% of flagged HCPs. Renewal talks begin in six weeks.

This PRD defines a prototype that (a) diagnoses why the current model does not change rep behaviour, (b) replaces it with an uplift (incremental response) model, (c) attaches a verified plain-language reason to every flagged HCP using the Mistral API, and (d) delivers a rep-facing view with territory filters. All data, model outputs, reasons, and metrics are stored in PostgreSQL.

## 2. Problem Statement

### 2.1 Observed facts

| Week | List opened | HCPs flagged | HCPs contacted | Contact rate |
| --- | --- | --- | --- | --- |
| 1 | Yes | 340 | 14 | 4.1% |
| 4 | Yes | 335 | 11 | 3.3% |
| 8 | Yes | 352 | 16 | 4.5% |

- Reps open the list every time, so awareness is not the problem.
- Contact rate is flat across weeks, so this is not a ramp-up issue.
- The Regional Field Trainer reports the list does not change who reps would call anyway.
- Validation covers only HCPs with 3 or more rep visits in the last 12 months.

### 2.2 Hypotheses to test

1. **Wrong target.** The model predicts who will prescribe, not who will prescribe more because of a visit. High prescribers score well regardless of rep effort.
2. **Selection bias.** Reps chose whom to visit. The validation universe (3+ visits) mixes "visited" with "likely to prescribe", so AUC overstates usefulness for the HCPs the list is meant to surface.
3. **No actionable reasons.** The list is a bare ranking with no explanation a rep can use in a call.
4. **Trust and workflow fit.** The list is not organised around territory or call-plan workflow.

## 3. Goals and Non-Goals

### 3.1 Goals

- Reproduce the current model and demonstrate the validation-universe bias and overlap with already-visited HCPs.
- Build an uplift model that ranks HCPs by incremental response to a visit and evaluate it with a Qini or uplift curve.
- Generate a short, factual reason per flagged HCP with Mistral, and verify every claim against that HCP's data.
- Deliver a rep-facing ranked list with reasons and territory filters.
- Define adoption and impact metrics and a randomized test design versus the current list.

### 3.2 Non-goals

- Real client data integration, CRM (Veeva) write-back, or production deployment.
- Authentication and role-based access (stubbed for the prototype).
- Patient-level data. Only HCP-level aggregates are used.
- Real-time scoring. Scoring is batch and weekly.

## 4. Users and Use Cases

| User | Need | Prototype feature |
| --- | --- | --- |
| Field rep | A short list of HCPs worth visiting this week, with a reason | Ranked list, reason text, territory filter |
| Regional Field Trainer | Evidence the list differs from habitual calls | "New vs. usual calls" badge, overlap metric |
| Sales Ops / analytics | Proof the new model is better and a way to test it | Qini curve, metrics page, test design |
| FDE / client lead | A defensible story before renewal talks | Diagnosis summary, deck |

## 5. Scope

### 5.1 In scope

1. Synthetic HCP universe generator (about 4,000 HCPs).
2. Current-model rebuild and diagnostics.
3. Uplift model and evaluation.
4. LLM reason generation and verification.
5. REST API and Next.js rep UI.
6. Adoption and impact metrics definition, plus A/B test design.
7. Optional deck of 8 slides or fewer.

### 5.2 Out of scope

Production infrastructure, SSO, mobile apps, multi-brand support, compliance (MLR) workflow tooling.

## 6. Functional Requirements

### FR1. Synthetic data generation

- Generate about 4,000 HCPs across derm and primary care specialties, multiple territories and regions, and assigned reps.
- Fields per HCP: specialty, territory, region, rep, 12-month baseline Rx, Rx trend, patient-volume proxy, past visit count, samples delivered, email engagement, last-visit recency.
- Baseline Rx drawn from distributions informed by public CMS Medicare Part D Prescribers by Provider summaries (derm and primary care). No real HCP identifiers.
- A hidden `true_uplift` per HCP drives a genuinely incremental visit response. It is highest for mid-volume derms with a rising trend and near zero for the very highest prescribers.
- Visit assignment is biased toward high prescribers to reproduce the real-world selection problem.
- A randomized subset (15 to 20% of HCPs) receives visits assigned at random, representing a pilot or random call blocks.
- Ground truth is stored in a separate table, never exposed to models or the API.
- Generation is seeded and reproducible.

### FR2. Current-model rebuild

- Train a gradient-boosted classifier predicting "increased Rx next quarter".
- Report AUC and top-decile precision on (a) the 3+ visit validation universe and (b) the full universe.
- Report feature importance and the overlap (percent) between the top decile and HCPs with heavy recent visits.
- Output a one-page diagnosis comparing the two universes.

### FR3. Uplift model

- Implement a T-learner (separate outcome models for visited and not-visited HCPs). An X-learner via causalml or econml is an optional comparison.
- Handle selection bias through propensity weighting and by evaluating on the randomized subset.
- Evaluate with a Qini curve and Qini coefficient on held-out HCPs, compared against the current model's ranking and a random baseline.
- Report rank correlation of predicted uplift against the planted `true_uplift`.
- Weekly scoring job writes scores to Postgres.

### FR4. Reasons with verification (Mistral API)

- For each flagged HCP, compute the top 3 drivers (SHAP) and collect the underlying field values.
- Call the Mistral API with a prompt that includes only those facts and instructs the model to write one or two plain sentences and to return structured claims (field, value).
- A verifier checks every claim against the HCP's stored row. A claim passes only if the field exists and the value matches within tolerance.
- On failure: regenerate once, then fall back to a deterministic template sentence.
- Store the reason text, claims, verification status, model name, prompt version, and token usage.
- Reasons never include uplift scores as promises of outcome, patient information, or off-label claims.

### FR5. Rep-facing application

- Ranked list with columns: rank, HCP, specialty, territory, uplift score band, top reason, "new vs. usual" badge.
- Filters: territory, region, specialty, rep. Sort by uplift or recency.
- HCP detail panel: drivers, verified reason, recent call history, baseline versus trend.
- Feedback control per HCP: "will call", "not relevant", with a reason code. Stored for adoption analysis.
- Metrics page: Qini curve, overlap diagnostics, verification pass rate.

### FR6. Adoption and impact tracking

- Log list opens, HCP detail views, feedback actions, and contacts from list.
- Compute adoption and impact metrics defined in Section 9.

## 7. Non-Functional Requirements

| Area | Requirement |
| --- | --- |
| Performance | List endpoint under 500 ms for 4,000 HCPs with filters. Full weekly scoring under 5 minutes. |
| Reproducibility | Seeded data generation and model training. Model and prompt versions stored with every output. |
| Privacy | Synthetic data only. Mistral receives only the minimal facts for one HCP at a time. |
| Reliability | Mistral failure never blocks the list. The template fallback is used. |
| Cost | Reason generation cached per HCP and week. Batch size and token budget logged. |
| Security | API keys in environment variables. Read-only API for the UI. Basic CORS rules. |
| Accessibility | Keyboard navigable table, sufficient contrast, readable on a laptop in the field. |

## 8. System Architecture

```
Generator → PostgreSQL ← Models (current, uplift) → Reasons job → PostgreSQL
                              ↑                          ↑
                        FastAPI (read API, feedback)   Mistral API
                              ↑
                        Next.js rep UI
```

### 8.1 Components

- **Data generator (Python):** writes the universe, visits, outcomes, and ground truth to Postgres.
- **Modelling jobs (Python):** scikit-learn or LightGBM for the current model, T-learner for uplift, SHAP for drivers, scikit-uplift or custom code for Qini.
- **Reasons job (Python):** calls Mistral, runs verification, writes results.
- **API (FastAPI):** read endpoints for the list, detail, territories, and metrics, plus a write endpoint for feedback and usage events.
- **Frontend (Next.js, React, TypeScript, Tailwind):** rep list, detail panel, metrics page.
- **Database (PostgreSQL):** single source of truth for all data.

### 8.2 Why this stack

- Python covers data, modelling, explainability, and the API in one language.
- PostgreSQL gives relational integrity, SQL for adoption analytics, and a path from prototype to production without a rewrite.
- Mistral API provides a low-cost hosted LLM with structured output, and an Ollama-hosted Mistral model is a drop-in option if data must stay local.
- Next.js delivers a fast, filterable UI and a clean static view for the HTML collateral.

## 9. Data Model (PostgreSQL)

| Table | Key columns |
| --- | --- |
| `hcp` | hcp\_id (PK), specialty, territory\_id, region, rep\_id, panel\_size\_proxy |
| `territory` | territory\_id (PK), name, region, rep\_id |
| `rx_history` | hcp\_id, period, trx, nbrx |
| `call_activity` | call\_id (PK), hcp\_id, rep\_id, call\_date, call\_type, samples, assigned\_randomly (bool) |
| `email_engagement` | hcp\_id, period, sent, opened, clicked |
| `ground_truth` | hcp\_id, true\_uplift (restricted, evaluation only) |
| `model_run` | run\_id (PK), model\_type, version, trained\_at, metrics\_json |
| `hcp_score` | run\_id, hcp\_id, week, score, uplift, rank, top\_drivers\_json |
| `hcp_reason` | hcp\_id, week, run\_id, reason\_text, claims\_json, verified (bool), fallback\_used (bool), llm\_model, prompt\_version, tokens\_in, tokens\_out |
| `rep_feedback` | feedback\_id (PK), hcp\_id, rep\_id, week, action, reason\_code, created\_at |
| `usage_event` | event\_id (PK), rep\_id, event\_type, hcp\_id, week, created\_at |

- Indexes on `hcp_score (week, rank)`, `hcp (territory_id)`, and `call_activity (hcp_id, call_date)`.
- The `ground_truth` table sits in a separate schema with no API access.

## 10. API Specification

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/hcps` | Ranked list. Query: territory, region, specialty, rep, week, limit, offset |
| GET | `/hcps/{id}` | Detail: drivers, reason, history |
| GET | `/territories` | Territory list for filters |
| GET | `/metrics` | Model comparison, Qini data, overlap, verification rate |
| POST | `/feedback` | Rep feedback on an HCP |
| POST | `/events` | Usage event logging |
| POST | `/admin/rescore` | Triggers scoring and reasons job (prototype only) |

### 10.1 Mistral integration

- Endpoint: Mistral chat completions API, with a JSON-schema constrained response.
- Model name, temperature (low, about 0.2), and prompt version are configuration values.
- Retries with exponential backoff, a request timeout, and a per-run token budget.
- Prompt contract: use only the supplied facts, no new numbers, no clinical or efficacy claims, maximum 40 words.

## 11. Evaluation and Success Metrics

### 11.1 Model metrics

| Metric | Target |
| --- | --- |
| Qini coefficient (uplift vs. current model) | Uplift model clearly above current model and random |
| Rank correlation with planted uplift | Spearman above 0.5 |
| Top-decile incremental Rx vs. current model's top decile | Higher on the randomized subset |
| Overlap of current model's top decile with heavily visited HCPs | Reported, expected to be high |
| Reason verification pass rate (first attempt) | 95% or above, 100% after fallback |

### 11.2 Adoption metrics (after rollout)

- **Contact rate on flagged HCPs:** contacted from list divided by flagged. Baseline is about 4%. Target is 25% or above within 8 weeks.
- **Active use:** share of reps who open HCP detail views and give feedback each week.
- **Top-N coverage:** share of each rep's top 20 contacted.
- **Feedback mix:** share of "will call" versus "not relevant" and the top reason codes.
- **Novelty:** share of contacted flagged HCPs the rep had not planned to call (new versus usual).

### 11.3 Impact metrics

- Incremental Rx per call on flagged versus control HCPs.
- NBRx and TRx lift in test versus control territories.
- Calls per incremental Rx (efficiency).

## 12. Test Design: New List vs. Current List

- **Unit:** territory (or rep) to limit contamination.
- **Assignment:** randomize matched territory pairs (by size, baseline Rx, specialty mix) into current-list and new-list arms.
- **Duration:** 6 to 8 weeks with a pre-period baseline.
- **Primary metric:** contact rate on flagged HCPs, and incremental Rx per call.
- **Analysis:** difference-in-differences on matched pairs, with confidence intervals.
- **Guardrails:** total call volume, rep satisfaction feedback, no drop in coverage of key accounts.
- **Fallback if territory randomization is not allowed:** within-rep holdout where a random 10% of eligible HCPs are withheld from the new list to measure lift.

## 13. Milestones

| Phase | Deliverable | Order |
| --- | --- | --- |
| 1 | Postgres schema, data generator, seeded universe | First |
| 2 | Current-model rebuild and diagnosis | Second |
| 3 | Uplift model, Qini evaluation | Third |
| 4 | Mistral reasons and verification | Fourth |
| 5 | FastAPI and Next.js UI | Fifth |
| 6 | Metrics page, test design, deck | Last |

If time is short, cut UI polish before evaluation rigor. The case is judged on diagnosis quality and proof the fix works.

## 14. Deliverables

- Runnable code repository (generator, models, reasons job, API, UI).
- Postgres schema and seed scripts.
- Rep-facing view with ranked list, reasons, and territory filter.
- Metrics output (Qini curve, overlap, verification rate).
- Optional deck, 8 slides or fewer.

## 15. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Synthetic data makes the uplift result look too convenient | Document the planted effect openly and test with noisy and weak-effect variants |
| LLM invents a fact in a reason | Structured claims, programmatic verification, template fallback |
| Mistral API outage or rate limits | Caching, retries, fallback reasons, optional local Ollama |
| Uplift estimates are noisy | Use randomized subset for evaluation, report confidence intervals, avoid overclaiming |
| Reps still ignore the list | Reasons, feedback loop, trainer involvement, A/B test with adoption metrics |
| Compliance concerns on promotional content | Reasons are internal call-planning aids, not HCP-facing, and exclude clinical claims |

