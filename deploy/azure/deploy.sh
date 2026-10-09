#!/usr/bin/env bash
# Deploy JobLens to Azure: Postgres Flexible Server, two Container Apps (API
# and dashboard, both scaling to zero) and a scheduled Container Apps Job for
# the daily ingest. Idempotent: re-running updates what exists.
#
#   IMAGE_TAG=preview-abc1234 bash deploy/azure/deploy.sh
#
# Read docs/deploy-azure.md first. This creates resources that bill your
# subscription. It never prints the database password: it is read from a
# hidden prompt (or DB_PASSWORD) and stored only as a Container Apps secret.
set -euo pipefail

: "${IMAGE_TAG:?set IMAGE_TAG, e.g. preview-abc1234 from the preview-image workflow}"
OWNER="${OWNER:-raviteja311}"
RG="${RG:-joblens-rg}"
LOCATION="${LOCATION:-centralindia}"
PG="${PG:-joblens-pg-$(printf %s "$OWNER" | tr -cd 'a-z0-9' | cut -c1-12)}"
ENVIRONMENT="${ENVIRONMENT:-joblens-env}"
API_APP="${API_APP:-joblens-api}"
UI_APP="${UI_APP:-joblens-ui}"
JOB="${JOB:-joblens-ingest}"
DB_ADMIN="${DB_ADMIN:-joblens}"
DB_NAME="${DB_NAME:-joblens}"
# 02:30 UTC is 08:00 in India.
INGEST_CRON="${INGEST_CRON:-30 2 * * *}"
INGEST_LIMIT="${INGEST_LIMIT:-2000}"
PYTHON="${PYTHON:-python}"

API_IMAGE="ghcr.io/${OWNER}/joblens:${IMAGE_TAG}"
UI_IMAGE="ghcr.io/${OWNER}/joblens-ui:${IMAGE_TAG}"

step() { printf '\n== %s\n' "$*"; }
exists() { "$@" >/dev/null 2>&1; }

az account show --query "{subscription:name, user:user.name}" -o table

if [ -z "${DB_PASSWORD:-}" ]; then
    read -rs -p "Postgres admin password (new server: 12+ chars, upper, lower, digit, symbol): " DB_PASSWORD
    echo
fi

step "resource group $RG in $LOCATION"
az group create -n "$RG" -l "$LOCATION" -o none

step "Postgres Flexible Server $PG (Burstable B1ms, 32 GB, PostgreSQL 16)"
if ! exists az postgres flexible-server show -g "$RG" -n "$PG"; then
    # --public-access 0.0.0.0 allows Azure services only (the apps and the
    # job), not the internet.
    az postgres flexible-server create -g "$RG" -n "$PG" -l "$LOCATION" \
        --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 \
        --admin-user "$DB_ADMIN" --admin-password "$DB_PASSWORD" \
        --public-access 0.0.0.0 --yes -o none
fi
# pgvector has to be allow-listed before `create extension vector` (migration
# 002) can run.
az postgres flexible-server parameter set -g "$RG" -s "$PG" \
    -n azure.extensions -v VECTOR -o none
if ! exists az postgres flexible-server db show -g "$RG" -s "$PG" -d "$DB_NAME"; then
    az postgres flexible-server db create -g "$RG" -s "$PG" -d "$DB_NAME" -o none
fi

HOST="$(az postgres flexible-server show -g "$RG" -n "$PG" --query fullyQualifiedDomainName -o tsv)"
DATABASE_URL="$("$PYTHON" -c 'import sys, urllib.parse as u; print(f"postgresql://{sys.argv[1]}:{u.quote(sys.argv[2], safe=str())}@{sys.argv[3]}:5432/{sys.argv[4]}?sslmode=require")' \
    "$DB_ADMIN" "$DB_PASSWORD" "$HOST" "$DB_NAME")"
unset DB_PASSWORD

step "Container Apps environment $ENVIRONMENT (no Log Analytics workspace)"
if ! exists az containerapp env show -g "$RG" -n "$ENVIRONMENT"; then
    az containerapp env create -g "$RG" -n "$ENVIRONMENT" -l "$LOCATION" \
        --logs-destination none -o none
fi

API_ENV=(DATABASE_URL=secretref:database-url LLM_BACKEND=none FORWARDED_ALLOW_IPS='*')

step "API $API_APP <- $API_IMAGE (scale 0 to 1, 1 vCPU, 2 GiB)"
if exists az containerapp show -g "$RG" -n "$API_APP"; then
    az containerapp secret set -g "$RG" -n "$API_APP" \
        --secrets "database-url=$DATABASE_URL" -o none
    az containerapp update -g "$RG" -n "$API_APP" --image "$API_IMAGE" \
        --set-env-vars "${API_ENV[@]}" -o none
else
    az containerapp create -g "$RG" -n "$API_APP" --environment "$ENVIRONMENT" \
        --image "$API_IMAGE" --target-port 8000 --ingress external \
        --min-replicas 0 --max-replicas 1 --cpu 1.0 --memory 2.0Gi \
        --secrets "database-url=$DATABASE_URL" --env-vars "${API_ENV[@]}" -o none
fi
API_URL="https://$(az containerapp show -g "$RG" -n "$API_APP" --query properties.configuration.ingress.fqdn -o tsv)"

step "dashboard $UI_APP <- $UI_IMAGE (scale 0 to 1, 0.25 vCPU, 0.5 GiB)"
if exists az containerapp show -g "$RG" -n "$UI_APP"; then
    az containerapp update -g "$RG" -n "$UI_APP" --image "$UI_IMAGE" \
        --set-env-vars "JOBLENS_API=$API_URL" -o none
else
    az containerapp create -g "$RG" -n "$UI_APP" --environment "$ENVIRONMENT" \
        --image "$UI_IMAGE" --target-port 8501 --ingress external \
        --min-replicas 0 --max-replicas 1 --cpu 0.25 --memory 0.5Gi \
        --env-vars "JOBLENS_API=$API_URL" -o none
fi
UI_URL="https://$(az containerapp show -g "$RG" -n "$UI_APP" --query properties.configuration.ingress.fqdn -o tsv)"

INGEST="python -m joblens migrate \
&& python -m joblens ingest --limit $INGEST_LIMIT \
&& python -m joblens embed --strategy whole \
&& python -m joblens embed --strategy section \
&& python -m joblens stats"

step "daily ingest job $JOB ($INGEST_CRON UTC)"
if exists az containerapp job show -g "$RG" -n "$JOB"; then
    az containerapp job secret set -g "$RG" -n "$JOB" \
        --secrets "database-url=$DATABASE_URL" -o none
    az containerapp job update -g "$RG" -n "$JOB" --image "$API_IMAGE" \
        --cron-expression "$INGEST_CRON" -o none
else
    az containerapp job create -g "$RG" -n "$JOB" --environment "$ENVIRONMENT" \
        --trigger-type Schedule --cron-expression "$INGEST_CRON" \
        --replica-timeout 3600 --replica-retry-limit 0 \
        --parallelism 1 --replica-completion-count 1 \
        --image "$API_IMAGE" --cpu 1.0 --memory 2.0Gi \
        --secrets "database-url=$DATABASE_URL" \
        --env-vars DATABASE_URL=secretref:database-url \
        --command sh --args -c "$INGEST" -o none
fi
unset DATABASE_URL

step "done"
echo "API:       $API_URL/health"
echo "dashboard: $UI_URL"
echo "First load: az containerapp job start -g $RG -n $JOB  (then see docs/deploy-azure.md)"
