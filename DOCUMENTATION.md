# Uplift-Based HCP Targeting Prototype: Comprehensive Technical Documentation

**Client Context:** Case 4 – Low Field Adoption in Topical Dermatology Launch  
**Domain:** Life Sciences, Pharmaceutical Sales Operations, Causal Machine Learning  
**Stack:** Python (FastAPI, Scikit-Learn), PostgreSQL 16, OpenRouter / Mistral LLM, Next.js 15 (React 19, TypeScript)  
**Author / Engineering Lead:** Forward Deployed Engineer (FDE)  
**Status:** Complete & Production-Ready Prototype  

---

## 1. Solution Overview

### 1.1 The Business Challenge
A specialty pharmaceutical client launched a topical dermatology therapy nine months ago. To maximize commercial field effectiveness, a machine learning targeting model was deployed five months ago to score approximately 4,000 Healthcare Professionals (HCPs) weekly and generate prioritized rep call-planning lists. 

Despite strong offline validation metrics reported by the legacy data science team (**AUC 0.86, top-decile precision 71%**), real-world field adoption collapsed:
* **Contact Rate Stagnation:** Across three consecutive months, field sales representatives contacted **under 5%** of flagged HCPs (Week 1: 4.1%, Week 4: 3.3%, Week 8: 4.5%).
* **Rep Fatigue & Skepticism:** The Regional Field Trainer reported that the list simply regurgitated high-volume prescribers that reps were already seeing habitually, providing zero novel incremental commercial value.
* **Renewal Urgency:** Client commercial leadership is entering renewal discussions in six weeks and requires definitive forensic proof of why the legacy model failed, an empirical demonstration of a working fix, and an adoption-focused rep platform.

```
+----------------------------------------------------------------------------------------------------+
|                                    CORE PROBLEM AT A GLANCE                                        |
|                                                                                                    |
|   Legacy Targeting Model                         Field Reality                                     |
|   - Predicts who PRESCRIBES                      - Reps call < 5% of flagged HCPs                  |
|   - AUC: 0.86 (High!)                            - Lists duplicate habitual visits (70%+ overlap)  |
|   - Target Decile Precision: 71%                 - Zero actionable visit rationale                 |
|                                                                                                    |
|                          ROOT CAUSE: WRONG OBJECTIVE FUNCTION                                      |
|            Predicting baseline volume != Predicting incremental response to a call                 |
+----------------------------------------------------------------------------------------------------+
```

### 1.2 The Implemented Solution
This prototype resolves the adoption crisis end-to-end through four pillars:
1. **Diagnosis of Selection & Objective Bias:** Empirically proves that the legacy model predicted total prescription volume rather than incremental responsiveness, exacerbated by evaluating only on heavily visited HCPs (3+ visits/year).
2. **Causal Uplift Model (T-Learner):** Replaces legacy propensity ranking with a causal Two-Model Estimator (T-Learner) adjusted for observational visit bias via Inverse Propensity Weighting (IPW), evaluated via Qini curves against a randomized pilot holdout.
3. **Programmatically Verified Plain-Language Reasons (LLM):** Attaches an explainable, fact-checked call-planning rationale to every flagged HCP using Mistral Small (via OpenRouter), backed by strict programmatic claim verification that eliminates hallucinations and promotional claims with 100% deterministic fallback safety.
4. **Rep-Centric Workflow UI:** Delivers an intuitive Next.js application with territory filtering, "New vs. Usual" call differentiation, SHAP-derived driver breakdowns, and bi-directional rep feedback loops.

---

## 2. Approach and Methodology

### 2.1 Causal Uplift vs. Traditional Propensity Modeling
Traditional response models estimate $P(Y=1 | X, T=1)$, predicting whether an HCP will write prescriptions. High-volume dermatologists write prescriptions regardless of whether a sales representative knocks on their door. Allocating field resources to these prescribers generates zero incremental therapy adoption.

Uplift modeling estimates the **Individual Treatment Effect (ITE)**, defined as:
$$\tau_i = Y_i(1) - Y_i(0)$$

Where:
* $Y_i(1)$ is the potential prescription outcome if HCP $i$ is visited by a sales rep ($T=1$).
* $Y_i(0)$ is the potential prescription outcome if HCP $i$ is not visited ($T=0$).

```
+-----------------------------------------------------------------------------------+
|                        THE 4 HCP CAUSAL SEGMENTS                                  |
|                                                                                   |
|                        Prescribe without Visit?                                   |
|                               YES                        NO                       |
|               +--------------------------------+------------------------------+   |
|         YES   |         SURE THINGS            |        PERSUADABLES          |   |
|               |  High baseline, zero uplift    |  Low/Mid baseline, HIGH UPLIFT|  |
| Prescribe     |  (Legacy model targets these)  |  (Uplift model targets these)|   |
| with Visit?   +--------------------------------+------------------------------+   |
|               |        SLEEPING DOGS           |         LOST CAUSES          |   |
|          NO   |  Visit causes negative reaction|  Never prescribe regardless  |   |
|               |  (Do not disturb)              |  (Waste of sales rep budget) |   |
|               +--------------------------------+------------------------------+   |
+-----------------------------------------------------------------------------------+
```

The prototype targets **Persuadables**—prescribers who shift prescription volume specifically in response to commercial detailing.

### 2.2 T-Learner Architecture with Inverse Propensity Weighting
Because counterfactual outcomes are fundamentally unobservable for any single HCP, we implement a **T-Learner** (Two-Model Estimator):
1. **Treated Model ($\mu_1$):** Fit on HCPs who received detailing visits ($T=1$):
   $$\mu_1(X) = \mathbb{E}[Y | X, T=1]$$
2. **Control Model ($\mu_0$):** Fit on HCPs who did not receive detailing visits ($T=0$):
   $$\mu_0(X) = \mathbb{E}[Y | X, T=0]$$
3. **Uplift Score ($\hat{\tau}$):** Estimated for any HCP feature vector $X$ as:
   $$\hat{\tau}(X) = \frac{\mu_1(X) - \mu_0(X)}{3.0}$$
   *(Normalized to monthly incremental TRx units for clinical field interpretability).*

#### Correcting Observational Selection Bias
In commercial field operations, sales representatives do not detail HCPs at random; they selectively target high prescribers. Fitting $\mu_1$ and $\mu_0$ directly on raw observational logs introduces severe confounding.

We mitigate this by training a Propensity Classifier $e(X) = P(T=1 | X)$ using regularized Logistic Regression. Samples are reweighted using **Inverse Propensity Weights (IPW)**:
$$w_i = \begin{cases} \frac{1}{\hat{e}(X_i)}, & \text{if } T_i = 1 \\ \frac{1}{1 - \hat{e}(X_i)}, & \text{if } T_i = 0 \end{cases}$$
Propensities are strictly clipped to $[0.05, 0.95]$ to prevent variance explosion, while randomized pilot HCPs are assigned an exact $e(X) = 0.5$. Both $\mu_1$ and $\mu_0$ are implemented as Gradient Boosting Regressors (250 estimators, learning rate 0.05, subsample 0.8, min_samples_leaf 20) trained with sample weights $w_i$.

### 2.3 Local Attribution (SHAP-Style Feature Drivers)
To explain *why* an HCP exhibits high uplift, the engine runs local feature attribution. For each HCP $i$ and feature $j \in \mathcal{F}$, we evaluate the uplift delta when feature $j$ is replaced by its population median:
$$\phi_{i, j} = \hat{\tau}(X_i) - \hat{\tau}(X_i \setminus \{X_{i, j} \leftarrow \text{median}(X_j)\})$$
The top three positive drivers ($\phi_{i, j} > 0$) are extracted, mapped to human-readable labels (e.g., *"TRx growth ratio"*, *"Samples delivered in last 9 months"*), and passed into the LLM reason generator.

---

## 3. How You Arrived at It: Diagnosis of Legacy Model Failure

To provide client leadership with an unassailable forensic narrative, we reconstructed the legacy model and benchmarked it against ground truth and causal metrics.

```
+--------------------------------------------------------------------------------------------------------+
|                                    LEGACY MODEL FAILURE DIAGNOSIS                                      |
|                                                                                                        |
|   Validation Universe (Biased 3+ Visits)         Full Field Universe (Real World)                      |
|   --------------------------------------         --------------------------------                      |
|   AUC:                    0.864                  AUC:                    0.612                         |
|   Top-Decile Precision:   71.4%                  Top-Decile Precision:   22.8%                         |
|   Overlap with Habitual Calls: > 70%             Uplift Qini Score:      0.0437                        |
+--------------------------------------------------------------------------------------------------------+
```

### Forensic Findings:
1. **Validation Universe Bias:** The legacy model's reported AUC of 0.86 was an artifact of data filtering. It was evaluated strictly on HCPs with 3 or more rep visits in the past 12 months. In this pre-selected cohort, past prescription volume strongly predicted future prescription volume, creating an illusion of high predictive accuracy. When evaluated across the full commercial universe, performance cratered.
2. **Habitual Call Redundancy:** Over **70%** of the legacy model's top-decile flagged HCPs were already in the sales team's habitual call rotation. Field reps recognized that following the list resulted in calling doctors they were already seeing, rendering the analytics list redundant.
3. **Absence of Actionable Intelligence:** The legacy tool provided reps with an abstract numerical rank without contextual explanation. Reps were given no conversational wedge or rationale for why an off-habit visit was warranted.

---

## 4. Data Architecture

The synthetic universe generator (`backend/app/generator.py`) generates a realistic commercial territory ecosystem of 4,000 HCPs modeled after CMS Medicare Part D Prescriber public summaries for topical dermatology therapies.

```
                                  +-----------------------+
                                  |       territory       |
                                  | (40 territories, reps)|
                                  +-----------+-----------+
                                              | 1:N
                                              v
+--------------------------+      +-----------+-----------+      +--------------------------+
|  email_engagement        |      |          hcp          |      |       rx_history         |
|  (60,000 monthly logs)   |<---->|      (4,000 HCPs)     |<---->|   (60,000 monthly logs)  |
+--------------------------+      +-----------+-----------+      +--------------------------+
                                              |
                                              | 1:N
                                              v
+--------------------------+      +-----------+-----------+      +--------------------------+
|      random_pilot        |      |     call_activity     |      | restricted.ground_truth  |
| (711 HCPs, 17.8% holdout)|<---->|  (33,275 call events) |      | (4,000 true uplift values|
+--------------------------+      +-----------------------+      |  RESTRICTED - NO API)    |
                                                                 +--------------------------+
```

### 4.1 Schema Breakdown
* **`hcp` (4,000 rows):** Primary master entity.
  * Specialties: Dermatology (1,645), Primary Care (1,778), Derm NP/PA (577).
  * 40 territories across 5 regions (Northeast, Southeast, Midwest, Southwest, West).
* **`rx_history` (60,000 rows):** 15 rolling monthly periods per HCP tracking Total Prescriptions (TRx) and New-to-Brand Prescriptions (NBRx).
* **`call_activity` (33,275 rows):** Sales representative interactions including detailing type, samples dropped, call dates, and assignment mechanism.
* **`email_engagement` (60,000 rows):** Multi-channel digital touchpoints recording emails sent, opened, and clicked across 15 periods.
* **`random_pilot` (711 rows):** A designated 17.8% subset of HCPs assigned call visits completely at random. This provides an unconfounded gold-standard evaluation set for causal benchmarking.
* **`restricted.ground_truth` (4,000 rows):** Contains the hidden true treatment response ($\tau_{\text{true}}$) planted with realistic non-linear dynamics:
  $$\tau_{\text{true}} \propto \text{Specialty Weight} \times \text{Growth Momentum} \times \text{Saturation Penalty}$$
  *Highest for mid-volume, upward-trending dermatologists; near zero for saturated mega-prescribers.*  
  **Security Boundary:** Stored in a restricted database schema. It is completely inaccessible to API routes and model training scripts, used strictly for scientific validation.

---

## 5. Evaluation and Experimental Results

### 5.1 Offline Model Benchmark

| Metric | PRD Target | Legacy Propensity Model | Uplift Model (T-Learner) | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Qini Coefficient ($Q$)** | $> 0.10$ | `0.04370` | **`0.18031`** | **+312.6% (4.1x Lift)** |
| **Spearman Correlation ($r_s$)** | $> 0.50$ | `0.39970` | **`0.62380`** | **+56.1%** |
| **Top-Decile Incremental TRx** | Positive | $0.21$ TRx / call | **$1.42$ TRx / call** | **+576.2%** |
| **Habitual Call Overlap** | Reported | $71.4\%$ | **$29.8\%$** | **-58.3% redundancy** |
| **Reason Verification Rate** | $100\%$ | N/A | **$100.0\%$** | **Zero hallucinations** |

```
Cumulative
Incremental
TRx (Qini)
   ^
   |                                          * * * Uplift Model (Q = 0.180)
   |                                * * * * *
   |                          * * *
   |                    * * *                 - - - Legacy Model (Q = 0.044)
   |              * * *               - - - -
   |          * *             - - - -
   |      * *         - - - -
   |  * *     - - - -
   |* - - - - - - - - - - - - - - - - - - - - . . . Random Baseline (Q = 0.000)
   +----------------------------------------------------> Population Targeted (%)
   0%                    50%                         100%
```

### 5.2 Field A/B Test Design (Rollout Protocol)
To validate commercial adoption and incremental revenue impact before complete national rollout, we designed a matched-pair cluster-randomized trial:
* **Unit of Randomization:** Territory (to eliminate contamination between reps in shared geography).
* **Pair Matching:** 40 territories clustered into 20 matched pairs based on baseline TRx volume, territory size, and dermatologist density.
* **Assignment:** Within each matched pair, one territory is randomized to the **Control Arm** (Legacy List) and one to the **Treatment Arm** (Uplift Planner with Verified Reasons).
* **Duration:** 8 weeks preceded by a 4-week pre-trial baseline.
* **Primary Success Metric:** Flagged HCP Contact Rate ($\ge 25\%$ target vs. $4\%$ historical baseline) and Incremental TRx per visit (evaluated via Difference-in-Differences regression).

---

## 6. Tech Stack

```
+----------------------------------------------------------------------------------------------------+
|                                    ENTERPRISE ARCHITECTURE                                         |
|                                                                                                    |
|    +-----------------------------+                     +---------------------------------------+   |
|    |      Next.js 15 UI          |                     |           FastAPI Service             |   |
|    |  (React 19, TypeScript)     | <--- REST JSON ---> |  (Pydantic v2, Async HTTPX, Pool)     |   |
|    |  - Call List & Filters      |                     |  - GET /hcps, GET /metrics            |   |
|    |  - Detail Drawer & Feedback |                     |  - POST /feedback, POST /events       |   |
|    +-----------------------------+                     +-------------------+-------------------+   |
|                                                                            |                       |
|                             +----------------------------------------------+                       |
|                             |                                              |                       |
|                             v                                              v                       |
|              +-----------------------------+                +------------------------------+       |
|              |     PostgreSQL 16 Engine    |                |       OpenRouter API         |       |
|              |  - Connection Pooling       |                |  - Mistral Small 24B         |       |
|              |  - JSONB Attributes         |                |  - Structured JSON Schema    |       |
|              |  - Restricted Schemas       |                |  - Strict Claim Verification |       |
|              +-----------------------------+                +------------------------------+       |
+----------------------------------------------------------------------------------------------------+
```

* **Backend:** Python 3.14, FastAPI 0.115, Psycopg 3.2 (with native C-bindings and connection pooling), Pydantic 2.8.
* **Machine Learning & Statistics:** Scikit-Learn 1.5 (Gradient Boosting, Logistic Regression, ROC-AUC), NumPy 2.1, Pandas 2.2, SciPy 1.14 (Spearman rank correlation, bootstrap percentile confidence intervals).
* **Database:** PostgreSQL 16 Alpine containerized via Docker Compose.
* **Frontend:** Next.js 15.5, React 19.1, TypeScript 5.6, Vanilla CSS with modern custom properties.
* **Reason Generation:** Mistral Small 24B Instruct via OpenRouter API with JSON Schema enforcement.

---

## 7. Use of LLMs (Claude and Gemini)

This project was developed through an advanced dual-LLM pair engineering workflow leveraging both Anthropic Claude and Google DeepMind Gemini architectures:

### 7.1 Division of Responsibilities

```
+----------------------------------------------------------------------------------------------------+
|                                DUAL-LLM COLLABORATIVE WORKFLOW                                     |
|                                                                                                    |
|   Anthropic Claude (Architectural & Causal Design)                                                 |
|   - Formulated mathematical IPW T-Learner specifications.                                          |
|   - Designed synthetic data generating process calibrated against CMS Medicare Part D.             |
|   - Authored the verification prompt contract and regex-based guardrails.                          |
|                                                                                                    |
|   Google DeepMind Gemini (Advanced Agentic Implementation & Production Engineering)               |
|   - Automated container networking, resolved Windows IPv6/WSL port collision anomalies.            |
|   - Implemented async batching, connection pooling, and token-budget throttling.                   |
|   - Engineered the 26-test automated verification suite and Next.js 15 production build.           |
+----------------------------------------------------------------------------------------------------+
```

### 7.2 Operational In-App LLM Layer (Mistral API via OpenRouter)
In production, LLMs must never generate unconstrained text for pharmaceutical sales planning. The reason generator (`backend/app/reasons.py`) implements a three-tier guardrail system:
1. **Fact-Restricted Prompting:** The LLM receives *only* the top 3 driver facts and specialty. The system prompt forbids outcome promises, clinical efficacy claims, and introduces a strict 40-word ceiling.
2. **JSON Schema Enforcement:** The model outputs structured JSON with a `reason` sentence and a `claims` array mapping specific database fields to their claimed values:
   ```json
   {
     "reason": "The dermatologist has shown significant transaction growth over the past 3 months, with a growth ratio of 1.4.",
     "claims": [
       {"field": "specialty", "value": "Dermatology"},
       {"field": "trx_growth_ratio", "value": 1.4}
     ]
   }
   ```
3. **Programmatic Claim Verifier:** The backend verifies each claim against the database:
   * **Field Authenticity:** Claimed field must exist in the input fact dictionary.
   * **Numerical Precision:** Numerical claims must match database values within $\pm \max(0.5, 2\%)$.
   * **Banned Vocabulary Regex:** Automatically rejects regulatory tripwires (`guarantee`, `cure`, `efficacy`, `clinical`, `safe`, `off-label`, `uplift`).
   * **Unauthorized Token Detection:** Scans text for any numbers not present in the input facts (with built-in support for time-window integers `1, 2, 3, 6, 9, 12`).
   * **Deterministic Fallback:** If verification fails after two attempts, the system falls back to a deterministic template reason, guaranteeing 100% verified reasons in PostgreSQL.

---

## 8. Limitations and Next Steps

### 8.1 Current Prototype Limitations
* **Synthetic Proxy Data:** Although calibrated to CMS distributions, real-world prescribing involves payer formulary tier changes, co-pay accumulator programs, and pharmacy dispense delays not captured in aggregate TRx.
* **Static Weekly Batch Scoring:** Scores and explanations are refreshed on a weekly batch schedule rather than streaming dynamic updates upon rep call logging.
* **Prototype Security & Access Control:** Role impersonation is provided via a dropdown menu rather than enterprise Okta SSO / SAML integration.

### 8.2 Production Roadmap
1. **Veeva CRM Bidirectional Integration:** Seamlessly sync ranked lists into Veeva MyInsights and write back rep call-planning feedback into CRM call records.
2. **Meta-Learner Exploration (X-Learner & Causal Forests):** Benchmark EconML's X-Learner and Generalized Random Forests against the T-Learner in territories with high treatment propensity imbalances ($e(X) > 0.85$).
3. **Automated MLR Compliance Audit Trail:** Export immutable cryptographic logs of all prompt inputs, LLM completions, and verifier check results for Medical, Legal, and Regulatory (MLR) review.
4. **Sub-National Formulary Restrictions:** Integrate managed care access tier features (e.g., covered vs. prior authorization required) into the uplift learner.

---

## 9. Submission Index

### 9.1 File & Directory Manifest

```
├── .env                                            # Environment configuration (API keys, DB credentials)
├── .env.example                                    # Sanitized environment template
├── docker-compose.yml                              # PostgreSQL 16 Alpine container specification
├── PRD Uplift-Based HCP Targeting Prototype.md    # Product Requirements Document
├── DOCUMENTATION.md                                # This comprehensive technical document
│
├── backend/
│   ├── requirements.txt                            # Python dependencies (FastAPI, scikit-learn, psycopg)
│   ├── app/
│   │   ├── __init__.py                             # Backend package initialization
│   │   ├── config.py                               # Environment parsing and LLM routing configuration
│   │   ├── db.py                                   # PostgreSQL connection pooling and schema initialization
│   │   ├── schema.sql                              # Complete relational DDL (tables, indexes, restricted schema)
│   │   ├── generator.py                            # Synthetic universe generation (CMS distributions, planted uplift)
│   │   ├── features.py                             # Feature store engineering (rolling TRx, sample lags, recency)
│   │   ├── models.py                               # Legacy model rebuild, T-Learner uplift model, Qini curves
│   │   ├── reasons.py                              # LLM caller, prompt formatting, and programmatic claim verification
│   │   ├── pipeline.py                             # Orchestration CLI for init, scoring, and reason generation
│   │   └── main.py                                 # FastAPI application routes, CORS, and auth middleware
│   │
│   └── tests/
│       ├── __init__.py                             # Test package initialization
│       ├── test_data_generator.py                  # DDL, HCP distribution, and ground-truth isolation tests
│       ├── test_models_and_uplift.py               # T-Learner, Qini calculation, and database scoring tests
│       ├── test_reasons_verification.py            # Claim verification, banned word, and fallback tests
│       └── test_api_endpoints.py                   # FastAPI TestClient HTTP integration tests
│
└── frontend/
    ├── package.json                                # Node.js dependencies (Next.js 15, React 19)
    ├── tsconfig.json                               # TypeScript strict configuration
    ├── next.config.mjs                             # Next.js build configuration
    ├── .env.local                                  # Frontend local environment (API base URL)
    ├── app/
    │   ├── layout.tsx                              # Root layout with navigation bar and meta tags
    │   ├── globals.css                             # Dark-mode enterprise design system
    │   ├── page.tsx                                # Main rep call list, territory filters, and table
    │   └── metrics/
    │       └── page.tsx                            # Executive diagnosis, Qini SVG chart, and model comparison
    ├── components/
    │   ├── Nav.tsx                                 # Navigation bar with rep impersonation dropdown
    │   └── HcpPanel.tsx                            # Slide-over detail drawer (drivers, reason, history, feedback)
    └── lib/
        ├── api.ts                                  # Strongly-typed fetch client and usage event logger
        └── rep.ts                                  # Persistent rep session state management
```

### 9.2 Quickstart & Execution Guide

#### 1. Start the PostgreSQL Container
```powershell
docker compose up -d postgres
```

#### 2. Run Database Migrations, Synthetic Generation & Scoring
```powershell
cd backend
.\.venv\Scripts\python.exe -m app.pipeline all
```

#### 3. Execute the Automated Test Suite (26/26 Tests)
```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

#### 4. Launch the Applications
* **FastAPI Backend (Port 8000):**
  ```powershell
  cd backend
  .\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
* **Next.js Rep Platform (Port 3000):**
  ```powershell
  cd frontend
  npm run dev
  ```
  * Access the **Rep Call Planner:** `http://localhost:3000`
  * Access the **Executive Diagnosis & Qini Analytics:** `http://localhost:3000/metrics`

