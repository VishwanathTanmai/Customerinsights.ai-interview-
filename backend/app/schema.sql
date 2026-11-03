-- Uplift-Based HCP Targeting schema (PRD section 9)
CREATE SCHEMA IF NOT EXISTS restricted;

CREATE TABLE IF NOT EXISTS territory (
    territory_id TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    region       TEXT NOT NULL,
    rep_id       TEXT NOT NULL,
    rep_name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hcp (
    hcp_id           TEXT PRIMARY KEY,
    display_name     TEXT NOT NULL,
    specialty        TEXT NOT NULL,
    territory_id     TEXT NOT NULL REFERENCES territory(territory_id),
    region           TEXT NOT NULL,
    rep_id           TEXT NOT NULL,
    panel_size_proxy INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_hcp_territory ON hcp(territory_id);

CREATE TABLE IF NOT EXISTS rx_history (
    hcp_id TEXT NOT NULL REFERENCES hcp(hcp_id),
    period INTEGER NOT NULL,         -- month index 1..15
    period_label TEXT NOT NULL,      -- YYYY-MM
    trx    INTEGER NOT NULL,
    nbrx   INTEGER NOT NULL,
    PRIMARY KEY (hcp_id, period)
);

CREATE TABLE IF NOT EXISTS call_activity (
    call_id           BIGSERIAL PRIMARY KEY,
    hcp_id            TEXT NOT NULL REFERENCES hcp(hcp_id),
    rep_id            TEXT NOT NULL,
    call_date         DATE NOT NULL,
    period            INTEGER NOT NULL,
    call_type         TEXT NOT NULL,
    samples           INTEGER NOT NULL,
    assigned_randomly BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_call_hcp_date ON call_activity(hcp_id, call_date);

CREATE TABLE IF NOT EXISTS email_engagement (
    hcp_id  TEXT NOT NULL REFERENCES hcp(hcp_id),
    period  INTEGER NOT NULL,
    sent    INTEGER NOT NULL,
    opened  INTEGER NOT NULL,
    clicked INTEGER NOT NULL,
    PRIMARY KEY (hcp_id, period)
);

-- Randomized pilot membership (observable: which HCPs were in random call blocks)
CREATE TABLE IF NOT EXISTS random_pilot (
    hcp_id TEXT PRIMARY KEY REFERENCES hcp(hcp_id)
);

-- Ground truth: restricted schema, never read by API or model training
CREATE TABLE IF NOT EXISTS restricted.ground_truth (
    hcp_id      TEXT PRIMARY KEY,
    true_uplift DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS model_run (
    run_id       BIGSERIAL PRIMARY KEY,
    model_type   TEXT NOT NULL,      -- current | uplift
    version      TEXT NOT NULL,
    week         TEXT NOT NULL,
    trained_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Feature snapshot used for scoring + reason verification
CREATE TABLE IF NOT EXISTS hcp_feature (
    hcp_id TEXT NOT NULL REFERENCES hcp(hcp_id),
    week   TEXT NOT NULL,
    features_json JSONB NOT NULL,
    PRIMARY KEY (hcp_id, week)
);

CREATE TABLE IF NOT EXISTS hcp_score (
    run_id BIGINT NOT NULL REFERENCES model_run(run_id) ON DELETE CASCADE,
    hcp_id TEXT NOT NULL REFERENCES hcp(hcp_id),
    week   TEXT NOT NULL,
    score  DOUBLE PRECISION NOT NULL,
    uplift DOUBLE PRECISION,
    rank   INTEGER NOT NULL,
    top_drivers_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (run_id, hcp_id)
);
CREATE INDEX IF NOT EXISTS ix_score_week_rank ON hcp_score(week, rank);

CREATE TABLE IF NOT EXISTS hcp_reason (
    hcp_id   TEXT NOT NULL REFERENCES hcp(hcp_id),
    week     TEXT NOT NULL,
    run_id   BIGINT NOT NULL REFERENCES model_run(run_id) ON DELETE CASCADE,
    reason_text TEXT NOT NULL,
    claims_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    verified BOOLEAN NOT NULL,
    first_attempt_verified BOOLEAN NOT NULL DEFAULT false,
    fallback_used BOOLEAN NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    llm_model TEXT,
    prompt_version TEXT,
    tokens_in  INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (hcp_id, week)
);

CREATE TABLE IF NOT EXISTS rep_feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    hcp_id TEXT NOT NULL REFERENCES hcp(hcp_id),
    rep_id TEXT NOT NULL,
    week   TEXT NOT NULL,
    action TEXT NOT NULL,
    reason_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_event (
    event_id BIGSERIAL PRIMARY KEY,
    rep_id   TEXT NOT NULL,
    event_type TEXT NOT NULL,
    hcp_id   TEXT,
    week     TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_status (
    job_id BIGSERIAL PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
