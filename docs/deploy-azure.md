# Deploying JobLens to Azure

The public demo: search and trends through the API and the Streamlit
dashboard, with a daily ingest of the Greenhouse, Lever and Ashby boards.
Chat is off (`LLM_BACKEND=none`): it needs Ollama, which Container Apps cannot
host, so it is shown in a recorded GIF instead.

| piece | Azure service | size |
| --- | --- | --- |
| database | Azure Database for PostgreSQL Flexible Server, PostgreSQL 16, `vector` allow-listed | Burstable B1ms, 32 GB |
| API | Container App `joblens-api`, image `ghcr.io/raviteja311/joblens` | 1 vCPU, 2 GiB, scale 0 to 1 |
| dashboard | Container App `joblens-ui`, image `ghcr.io/raviteja311/joblens-ui` | 0.25 vCPU, 0.5 GiB, scale 0 to 1 |
| daily ingest | Container Apps Job `joblens-ingest`, same image as the API | 1 vCPU, 2 GiB, 02:30 UTC (08:00 IST) |

Everything lives in one resource group, `joblens-rg`, in Central India, so
deleting the group deletes the deployment.

## Cost (Azure for Students)

- The $100 student credit pays for everything here. Whether the free
  12-month Postgres B1ms grant applies to a student subscription could not be
  confirmed; assume Postgres is paid, roughly $15 to $20 a month in Central
  India, so the credit lasts about five months.
- Container Apps has a monthly free grant (180,000 vCPU-seconds, 360,000
  GiB-seconds). It only covers this setup because both apps scale to zero.
  The daily job uses about 27,000 vCPU-seconds a month.
- No Log Analytics workspace is created (`--logs-destination none`), so there
  is no log ingestion bill. The trade-off: no stored logs, only live log
  streaming.
- To save credit when the demo is not needed, stop the database:
  `az postgres flexible-server stop -g joblens-rg -n <server>`. Azure starts a
  stopped server again automatically after seven days.

**Cold starts.** The API loads its embedding model and cross-encoder at
start-up. After it has scaled to zero, the first request waits about 20 to 40
seconds; requests after that are normal.

## One-time setup (yours)

1. Activate **Azure for Students** with your university email.
2. Install the **Azure CLI** (`winget install Microsoft.AzureCLI`) and open a
   new terminal.
3. `az login`, in a browser, yourself. Check the subscription with
   `az account show`.
4. Register the providers once:
   `az provider register -n Microsoft.App` and
   `az provider register -n Microsoft.DBforPostgreSQL`.

## 1. Build the images

Run the **preview-image** workflow on the branch to deploy:

```bash
gh workflow run preview-image --ref india-corpus
```

It runs the tests and pushes `joblens:preview-<sha>` and
`joblens-ui:preview-<sha>` to GitHub Container Registry; the tag is in the
run summary. The first time only, make both packages public (GitHub, your
profile, Packages, each package, Package settings, Change visibility), or
Container Apps cannot pull them.

`latest` is still built only from `main`, by `deploy.yml`, behind the tests
and the retrieval eval gate.

## 2. Create or update the deployment

From the repo root, in Git Bash:

```bash
IMAGE_TAG=preview-<sha> bash deploy/azure/deploy.sh
```

It asks for the Postgres admin password at a hidden prompt (Azure needs 12+
characters with upper and lower case, a digit and a symbol). Keep it in a
password manager. It is stored only as a Container Apps secret and never
printed. Re-running the script with a new tag updates the apps and the job.

## 3. First load and checks

The database starts empty. Run the ingest job once by hand (about 15 minutes:
migrations, all 291 boards, both embedding indexes):

```bash
az containerapp job start -g joblens-rg -n joblens-ingest
az containerapp job execution list -g joblens-rg -n joblens-ingest -o table
```

Then check, in order:

1. **pgvector is 0.8 or newer.** Open `https://<api>/health` and read
   `pgvector`. `vector_search` sets `hnsw.iterative_scan`, which older
   versions reject, and there is an [Azure Q&A thread about Flexible Server
   crashing with vector
   0.8.0](https://learn.microsoft.com/en-us/answers/questions/2284930/azure-database-for-postgresql-flexible-server-cras),
   so also run a section-strategy search (check 3) before trusting it.
2. The same `/health` reports about 700 postings and 5,600 chunks.
3. The dashboard loads, a search for "data engineer Pune" returns Pune
   postings, and a section-strategy search returns a full page (the
   iterative-scan fix working on Azure's pgvector).

## Updating

New code: run the preview-image workflow, then the script with the new tag.
The job picks up the new image on its next run.

## Tearing down

```bash
az group delete -n joblens-rg
```

This permanently deletes the database, both apps and the job. Export anything
you want first.
