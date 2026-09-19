-- Phase 4: every LLM call, logged.
--
-- Written before the first endpoint, not after, because the questions this
-- table answers ("why did that answer change", "what is this costing",
-- "which prompt version was live on Tuesday") cannot be answered
-- retroactively. The prompt and the response are stored in full: they are
-- the only evidence of what the model was actually asked.

create table if not exists llm_calls(
    id bigserial primary key,
    created_at timestamptz not null default now(),
    feature text not null,
    backend text not null,
    model text not null,
    prompt_name text,
    prompt_version text,
    prompt text not null,
    response text,
    prompt_tokens integer,
    completion_tokens integer,
    cost_usd numeric(10, 6) default 0,
    latency_ms integer,
    ok boolean not null default true,
    -- More than one means the first response failed schema validation and
    -- was retried. A rising average here is a prompt regression.
    attempts integer not null default 1,
    error text
);

create index if not exists llm_calls_created_idx on llm_calls(created_at desc);
create index if not exists llm_calls_feature_idx on llm_calls(feature, created_at desc);


-- Phase 5: one row per eval run, so the README can show a trend rather than
-- a single number that was true once.

create table if not exists eval_runs(
    id bigserial primary key,
    created_at timestamptz not null default now(),
    suite text not null,
    config text not null,
    git_sha text,
    queries integer not null,
    metrics jsonb not null,
    note text
);

create index if not exists eval_runs_suite_idx on eval_runs(suite, created_at desc);
