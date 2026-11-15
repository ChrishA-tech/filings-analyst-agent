#!/usr/bin/env bash
# Deploy the analyst API to Azure Container Apps, backed by Azure OpenAI.
# Free path: Azure for Students ($100 credit, no card). Delete the resource group when done: see the last line.
set -euo pipefail
RG=${RG:-filings-analyst-rg}
LOC=${LOC:-swedencentral}          # pick a region where your subscription can deploy the models below
AOAI=${AOAI:-filings-aoai-$RANDOM}
CHAT=${AZURE_OPENAI_CHAT:-gpt-4.1-mini}
EMBED=${AZURE_OPENAI_EMBED:-text-embedding-3-small}

az group create -n "$RG" -l "$LOC" -o none
az cognitiveservices account create -n "$AOAI" -g "$RG" -l "$LOC" --kind OpenAI --sku S0 --custom-domain "$AOAI" --yes -o none
deploy() {  # name version
  az cognitiveservices account deployment create -g "$RG" -n "$AOAI" --deployment-name "$1" --model-name "$1" \
    --model-version "$2" --model-format OpenAI --sku-name GlobalStandard --sku-capacity 20 -o none \
  || az cognitiveservices account deployment create -g "$RG" -n "$AOAI" --deployment-name "$1" --model-name "$1" \
    --model-version "$2" --model-format OpenAI --sku-name Standard --sku-capacity 20 -o none
}
deploy "$CHAT" "${CHAT_VERSION:-2025-04-14}"
deploy "$EMBED" "${EMBED_VERSION:-1}"
ENDPOINT=$(az cognitiveservices account show -n "$AOAI" -g "$RG" --query properties.endpoint -o tsv)
KEY=$(az cognitiveservices account keys list -n "$AOAI" -g "$RG" --query key1 -o tsv)

cat > .env.azure <<ENV
export AZURE_OPENAI_ENDPOINT=$ENDPOINT
export AZURE_OPENAI_API_KEY=$KEY
export AZURE_OPENAI_CHAT=$CHAT
export AZURE_OPENAI_EMBED=$EMBED
ENV
echo "Wrote .env.azure (git-ignored). Evals: source .env.azure && python run_evals.py --backend azure"

if [[ "${DEPLOY_API:-1}" == "1" ]]; then
  az extension add --name containerapp --upgrade -y -o none
  az containerapp up -n filings-analyst -g "$RG" -l "$LOC" --source . --ingress external --target-port 8000 \
    --env-vars ANALYST_BACKEND=azure AZURE_OPENAI_ENDPOINT="$ENDPOINT" AZURE_OPENAI_CHAT="$CHAT" AZURE_OPENAI_EMBED="$EMBED" -o none
  az containerapp secret set -n filings-analyst -g "$RG" --secrets aoai-key="$KEY" -o none
  az containerapp update -n filings-analyst -g "$RG" --set-env-vars AZURE_OPENAI_API_KEY=secretref:aoai-key -o none
  URL=$(az containerapp show -n filings-analyst -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
  echo "API: https://$URL/health   (POST /ask)"
fi
echo "When finished, stop spending credit:  az group delete -n $RG --yes --no-wait"
