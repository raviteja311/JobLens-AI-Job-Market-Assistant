# Labelling guide: the retrieval golden set

How every judgement in `data/golden/retrieval.yaml` is graded. The point of
writing it down is consistency: the same posting should get the same grade
for the same query on a Monday and on a Friday, and a second grader reading
this page should mostly agree. Self-agreement is measured (see the end), so
a vague rule shows up as a low kappa.

Drafted by Claude Opus 5.5 on 2026-10-08 for Jetti Raviteja to edit and
adopt. Change any rule you disagree with **before** grading starts, never
halfway through.

## The three grades

| Grade | Meaning | Test |
| --- | --- | --- |
| **2** | Exactly what was asked | The person who typed the query would apply. Every constraint in the query is met by what the posting says. |
| **1** | Acceptable | The person would open it and consider it. The role family is right, but one constraint is missed by one step or not stated. |
| **0** | Not relevant | The person would skip it. Wrong role family, or a constraint missed outright. |

Grade the posting, not the search engine. Do not look at which retriever
found a candidate (`found by:` in the pool) before deciding.

## General rules

1. **Read the posting, not the title.** Titles mislead in both directions:
   "Associate Manager - Data Engineer" is a manager role, "Analyst II, Data
   Science" is a data scientist role, "Senior Software Engineer - AI" may be
   a backend role that calls an LLM API. Decide from the responsibilities
   and requirements.
2. **Absence is not evidence.** If the query asks for something the posting
   does not mention (a team size, an interview format, a tool), the posting
   has not met that constraint. It can still be a 1 if everything else fits;
   it cannot be a 2.
3. **One constraint missed by one step is a 1. Missed outright, or two
   missed, is a 0.** "One step" is defined per constraint below.
4. **When torn between two grades, give the lower one** and move on. A
   consistent strict grader is worth more than an inconsistent generous one.
5. **Duplicates get the same grade.** The same role posted twice (same
   company and title, different requisition id, for example Brillio's
   "Lead AI Engineer I" series) is graded identically.
6. **Talent pools and generic listings** ("Future Opportunities",
   "Expression of Interest") are at most a 1: there is no concrete role to
   apply to.
7. **Time-box it.** About 30 seconds per candidate. If a posting needs more
   than a minute to decide, give the lower grade and write the query id in
   your notes so the rule can be sharpened.

## Constraints, and what "one step" means

### Role family

The family the query names: data analyst, data engineer, data scientist, ML
engineer, MLOps, AI/LLM engineer, BI developer, researcher, analytics
manager.

- Same family: meets it.
- Adjacent family (analyst and BI developer, data scientist and ML engineer,
  ML engineer and MLOps, AI engineer and ML engineer): one step.
- Anything else, including a software engineer whose job only touches data
  or AI in passing: missed outright.
- "AI" or "data" in a title is not enough. A support, sales, security,
  programme management or content role with "AI" in the title is a 0 for
  every role query, however it was retrieved.

### Experience level

The ladder: **intern/fresher (0-1 years) → junior (1-3) → mid (3-5) →
senior (5-8) → lead/staff/principal (8+) → head/director.**

- Use the years the posting states. Use the title only if no years are
  given, and remember "Associate" means junior at some companies (Beghou)
  and a step above senior at others (Sigmoid's "Associate Lead").
- A range in the query ("2-4 years") is met if the posting's required range
  overlaps it.
- One rung away: one step. Two or more: missed outright. A fresher query
  against a senior posting is always 0.
- "Fresher" or "entry level" is a 2 only if the posting says 0-1 or 0-2
  years, names freshers or new graduates, or is an internship.

### Location

- A city in the query ("in Hyderabad") is met by a posting in that city,
  including multi-city postings that list it ("Bangalore; Pune; Gurgaon"
  meets "Pune").
- A different Indian city: one step. Remote within India: one step.
- "India" in the query is met by any Indian location or remote-in-India.
- A posting whose only locations are outside India: missed outright (these
  should not be in the corpus; note the key if you see one).

### Work mode

- "Remote" is met only if the posting says remote and open to India.
  "Remote" with a different country, or "hybrid": one step. Onsite: missed.
- "Hybrid" is met by hybrid. Onsite or fully remote: one step.
- "Onsite" is met by onsite or an office location with no remote mention.
- "Work from home" means remote.
- "US shift" or "US time zone" is met only if the posting states overlap
  with US hours. Remote without hours stated: one step.

### Skills and tools

- Met if the named skills are central: in the requirements or the
  day-to-day responsibilities.
- A query naming several tools ("Airflow dbt Snowflake") is met if most are
  central and the rest appear. Only one of them, or all only as
  nice-to-have: one step.
- None of them: missed outright, even if the role family is right.

### Company type and domain

- "Fintech", "healthcare", "insurance", "e-commerce" and similar are met if
  the company's business is in that domain **or** the role's own work is
  (an insurance analytics team at a consulting firm meets "insurance
  analytics").
- "Startup" is met if the posting or company describes itself as one, or is
  clearly early stage. Not stated: one step.
- "Product company" vs "service company": a company selling its own product
  vs one doing client work (consulting, outsourcing, staffing). Not
  determinable from the posting: one step.

### Negations ("not cloud-heavy", "Azure not AWS", "without DSA rounds")

- Met only if the posting satisfies the positive part **and** shows the
  negated thing is absent or minor (for example Azure required and AWS not
  mentioned, or AWS listed only as a nice-to-have).
- The negated thing is a core requirement: missed outright.
- The posting gives no evidence either way (no posting describes its
  interview rounds): the negation is not met, so the best possible grade is
  1. It is expected that some hard queries have no 2 at all. That is a true
  fact about the corpus, and the eval should reward a retriever that ranks
  the 1s first, not one that invents 2s.

### Intent queries ("analyst wanting to move into ML", "ML role that is not just dashboards")

Grade the role against the intent, not the words. "Analyst wanting to move
into ML" is a 2 for an analyst or junior data scientist role with real
modelling work and a level the analyst can get; a senior ML engineer role
is a 0, however well it matches the words.

## Worked examples from the pool

| Query | Posting | Grade | Why |
| --- | --- | --- | --- |
| fresher data analyst job in Hyderabad | Senior MDM Product Data Analyst, Pure Storage, Bangalore | 0 | Senior vs fresher is three rungs, and the city is wrong too. |
| data analyst internship | Data Analyst Intern, Portcast (Bangalore, Chennai, Singapore and others) | 2 | Intern, analyst, and Indian cities among its locations. |
| junior machine learning engineer Bengaluru | Sr. Machine Learning Engineer, Conga, Bangalore | 0 | Senior vs junior is two rungs. |
| associate consultant analytics for freshers | Associate Consultant- Advanced Analytics (I0048), Beghou, Bangalore | 2 or 1 | 2 if it states 0-2 years or new graduates; 1 if it asks for 2-3 years. |
| quant researcher at a trading firm | Quantitative Researcher (2027 Graduate), Graviton Research Capital, Gurugram | 2 | Exact role and firm type. |
| AI for cybersecurity | Senior Technical Consultant - Forward Deployed AI Security Engineer, AHEAD, Gurugram | 2 or 1 | 2 if the work is applying AI to security; 1 if it is securing AI systems. Decide once and apply it to every similar posting. |
| MLOps engineer India | Full Stack Engineering Lead - AI & GenAI, WPP, Chennai | 0 | Wrong family: full-stack lead. |

## Recording grades

One JSON object per batch, query id to `{"source:source_id": grade}`, using
the key printed after `---` in the pool file. Grade every candidate in a
query's pool, zeros included: "judged and rejected" and "never looked at"
are different states. Merge with:

```bash
python scripts/golden_apply.py grades.json --grader human --note "graded by Jetti Raviteja on <date> from full posting text"
```

Only `--grader human` can mark a query verified, and only the verified
queries are scored by the eval.

## Measuring consistency

1. At least three days after the first pass, pick 10 queries at random
   (`random.sample` over the ids, seed recorded).
2. Re-grade their pools blind: work from a fresh pool dump, not the YAML.
3. Compute Cohen's kappa between the two passes over every candidate in
   those 10 queries (`sklearn.metrics.cohen_kappa_score`, with
   `weights="linear"` because the grades are ordinal). Report the number
   and the candidate count in the README.
4. Look at every disagreement. If several share a cause, the rule above is
   unclear: sharpen it here, note the change and its date at the bottom of
   this page, and re-grade the queries it affects.

Rough reading: above 0.8 is strong, 0.6 to 0.8 is substantial, below 0.6
means a rule needs work before the numbers are worth quoting.

## Changes to this guide

- 2026-10-08: first version.
