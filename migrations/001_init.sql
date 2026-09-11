-- Phase 1 schema.
--
-- Two layers. raw_postings keeps whatever the source gave us, untouched.
-- postings holds our interpretation of it. When the salary parser turns out
-- to be wrong (it will), we fix it and re-run the transform over raw_postings
-- instead of waiting two weeks to re-collect the data.



create table if not exists raw_postings(
    id bigserial primary key,
    source text not null,
    source_id text not null,
    payload jsonb not null,
    fetched_at timestamptz not null default now(),
    run_id uuid not null
);


-- One row per (source, posting) per fetch. We keep the history rather than
-- overwriting, so we can see when a posting's text changed.

create index if not exists raw_postings_source_idx on raw_postings(source, source_id);
create index if not exists raw_postings_run_idx on raw_postings(run_id);

create table if not exists postings(
    id bigserial primary key,
    source text not null,
    source_id text not null,
    title text not null,
    company text not null,
    location text,
    is_remote boolean not null default false,
    salary_raw text,
    salary_min numeric,
    salary_max numeric,
    salary_period text,
    salary_currency text,
    salary_min_year numeric,
    salary_max_year numeric,
    description text,
    url text not null,
    posted_at timestamptz,
    content_hash text not null,
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    unique(source, source_id)
);

create index if not exists postings_hash_idx on postings(content_hash);
create index if not exists postings_posted_at_idx on postings(posted_at desc);

create table if not exists ingestion_runs(
    run_id uuid primary key,
    source text not null,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    fetched integer not null default 0,
    inserted integer not null default 0,
    updated integer not null default 0,
    duplicates integer not null default 0,
    status text not null default 'running',
    error text
);

create index if not exists ingestion_runs_started_idx on ingestion_runs(started_at desc);