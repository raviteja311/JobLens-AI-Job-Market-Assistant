-- Phase 3 schema: full-text search and vector search on the same postings.

create extension if not exists vector;


-- Keyword half of hybrid retrieval.
--
-- Generated and stored rather than computed per query, because the corpus is
-- written once a day and read on every search. The weights say that a match
-- in the title counts for more than a match three paragraphs into the
-- description, which is the one piece of ranking knowledge Postgres needs
-- from us.

alter table postings
    add column if not exists search_vector tsvector
    generated always as (
        setweight(to_tsvector('english', coalesce(title, '')), 'A')
        || setweight(to_tsvector('english', coalesce(company, '')), 'B')
        || setweight(to_tsvector('english', coalesce(description, '')), 'C')
    ) stored;

create index if not exists postings_search_idx on postings using gin(search_vector);


-- Vector half.
--
-- One row per chunk, not per posting: the chunking strategy is a variable we
-- measure, so both strategies coexist and a query picks one. `model` is in
-- the key for the same reason.
--
-- The dimension is pinned to 384 (all-MiniLM-L6-v2) because pgvector needs a
-- fixed width to build an index. Comparing a model with a different width is
-- done in the eval harness in memory rather than by widening this table; when
-- one wins, it gets migration 003 and this column changes.

create table if not exists posting_chunks(
    id bigserial primary key,
    posting_id bigint not null references postings(id) on delete cascade,
    strategy text not null,
    chunk_index integer not null,
    content text not null,
    model text not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now(),
    unique(posting_id, strategy, model, chunk_index)
);

create index if not exists posting_chunks_posting_idx on posting_chunks(posting_id);
create index if not exists posting_chunks_lookup_idx on posting_chunks(strategy, model);

-- HNSW over cosine distance. Built on the whole table rather than per
-- (strategy, model) because pgvector cannot index a subset; the lookup index
-- above carries the filter and the planner combines them.
create index if not exists posting_chunks_embedding_idx
    on posting_chunks using hnsw (embedding vector_cosine_ops);


-- Phase 1 dedup was an exact hash. Phase 3 adds a near-duplicate verdict from
-- embedding similarity, kept in its own table so the hash baseline stays
-- intact and the two can be compared.

create table if not exists posting_duplicates(
    posting_id bigint not null references postings(id) on delete cascade,
    duplicate_of bigint not null references postings(id) on delete cascade,
    similarity real not null,
    method text not null,
    detected_at timestamptz not null default now(),
    primary key (posting_id, duplicate_of, method)
);
