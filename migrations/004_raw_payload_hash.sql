-- Bronze stores a posting only when its payload changed (db.insert_raw).
-- Sending every payload to Postgres to compare it took almost two minutes
-- for 24,000 board jobs, nearly all of it the transfer. So the client sends
-- a hash of each payload first, and only the changed payloads after it.
--
-- payload_hash is computed in Python (db.payload_hash), not generated here,
-- because the point is to avoid sending the payload at all. Rows from before
-- this migration have no hash and count as changed once, which costs one
-- extra copy of each and is otherwise harmless.

alter table raw_postings add column if not exists payload_hash text;

create index if not exists raw_postings_latest_idx
    on raw_postings(source, source_id, fetched_at desc, id desc)
    include (payload_hash);
