-- Put the posting's location into keyword search.
--
-- 002 built search_vector from title, company and description only, so a
-- city was searchable only when the description happened to repeat it: 28 of
-- the 93 Pune postings do. "data engineer jobs in Pune" put one Pune posting
-- in the top ten. Location goes in at weight B, the same as company: it is a
-- short field the posting states on purpose, not a word in passing.
--
-- A generated column cannot be altered in place, so it is dropped and added
-- again, which also rebuilds its GIN index. The check makes this run once:
-- migrate() runs every file on every start.

do $$
begin
    if not exists (
        select 1
          from information_schema.columns
         where table_name = 'postings'
           and column_name = 'search_vector'
           and generation_expression like '%location%'
    ) then
        drop index if exists postings_search_idx;
        alter table postings drop column if exists search_vector;
        alter table postings
            add column search_vector tsvector
            generated always as (
                setweight(to_tsvector('english', coalesce(title, '')), 'A')
                || setweight(to_tsvector('english', coalesce(company, '')), 'B')
                || setweight(to_tsvector('english', coalesce(location, '')), 'B')
                || setweight(to_tsvector('english', coalesce(description, '')), 'C')
            ) stored;
        create index postings_search_idx on postings using gin(search_vector);
    end if;
end
$$;
