# Azure Deployment Guide

This guide covers deploying the Agentic Post Generator to Azure using either Container Apps or App Service.

## Prerequisites

- **Azure Subscription**
- **Azure CLI** installed and logged in
- **Docker** installed locally
- **Azure OpenAI** deployment with chat and embeddings models
- **Azure Cosmos DB** account (optional, can use in-memory for testing)

## Option 1: Azure Container Apps (Recommended)

Container Apps provides auto-scaling, managed ingress, and revision management.

### 1. Create Resource Group

```bash
az group create \
  --name rg-agentic-post-gen \
  --location eastus
```

### 2. Create Container Registry

```bash
az acr create \
  --resource-group rg-agentic-post-gen \
  --name acragenticpostgen \
  --sku Basic
```

### 3. Build and Push Docker Image

```bash
# Login to ACR
az acr login --name acragenticpostgen

# Build and push
docker build -t acragenticpostgen.azurecr.io/post-generator:latest .
docker push acragenticpostgen.azurecr.io/post-generator:latest
```

### 4. Create Container App Environment

```bash
az containerapp env create \
  --name aca-env-post-gen \
  --resource-group rg-agentic-post-gen \
  --location eastus
```

### 5. Create Container App

```bash
az containerapp create \
  --name aca-post-generator \
  --resource-group rg-agentic-post-gen \
  --environment aca-env-post-gen \
  --image acragenticpostgen.azurecr.io/post-generator:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server acragenticpostgen.azurecr.io \
  --registry-identity system \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --env-vars \
    "AZURE_OPENAI_ENDPOINT=secretref:azure-openai-endpoint" \
    "AZURE_OPENAI_API_KEY=secretref:azure-openai-key" \
    "AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4" \
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002" \
    "CHECKPOINTER=cosmos" \
    "COSMOS_ENDPOINT=secretref:cosmos-endpoint" \
    "COSMOS_KEY=secretref:cosmos-key"
```

### 6. Set Secrets

```bash
az containerapp secret set \
  --name aca-post-generator \
  --resource-group rg-agentic-post-gen \
  --secrets \
    azure-openai-endpoint="<YOUR_AZURE_OPENAI_ENDPOINT>" \
    azure-openai-key="<YOUR_AZURE_OPENAI_KEY>" \
    cosmos-endpoint="<YOUR_COSMOS_ENDPOINT>" \
    cosmos-key="<YOUR_COSMOS_KEY>"
```

### 7. Get App URL

```bash
az containerapp show \
  --name aca-post-generator \
  --resource-group rg-agentic-post-gen \
  --query properties.configuration.ingress.fqdn
```

## Option 2: Azure App Service

App Service provides a simpler deployment model with fixed capacity.

### 1. Create App Service Plan

```bash
az appservice plan create \
  --name asp-post-generator \
  --resource-group rg-agentic-post-gen \
  --is-linux \
  --sku B1
```

### 2. Create Web App

```bash
az webapp create \
  --resource-group rg-agentic-post-gen \
  --plan asp-post-generator \
  --name app-agentic-post-gen \
  --deployment-container-image-name acragenticpostgen.azurecr.io/post-generator:latest
```

### 3. Configure App Settings

```bash
az webapp config appsettings set \
  --name app-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --settings \
    AZURE_OPENAI_ENDPOINT="<YOUR_AZURE_OPENAI_ENDPOINT>" \
    AZURE_OPENAI_API_KEY="<YOUR_AZURE_OPENAI_KEY>" \
    AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4" \
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT="text-embedding-ada-002" \
    CHECKPOINTER="cosmos" \
    COSMOS_ENDPOINT="<YOUR_COSMOS_ENDPOINT>" \
    COSMOS_KEY="<YOUR_COSMOS_KEY>" \
    WEBSITES_PORT="8000"
```

### 4. Enable Container Registry Access

```bash
az webapp config container set \
  --name app-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --docker-registry-server-url https://acragenticpostgen.azurecr.io \
  --docker-registry-server-user $(az acr credential show -n acragenticpostgen --query username -o tsv) \
  --docker-registry-server-password $(az acr credential show -n acragenticpostgen --query passwords[0].value -o tsv)
```

## Setup Azure Cosmos DB

### 1. Create Cosmos DB Account

```bash
az cosmosdb create \
  --name cosmos-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --default-consistency-level Session
```

### 2. Create Database

```bash
az cosmosdb sql database create \
  --account-name cosmos-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --name agentic_post_gen
```

### 3. Create LTM Container

```bash
az cosmosdb sql container create \
  --account-name cosmos-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --database-name agentic_post_gen \
  --name ltm \
  --partition-key-path "/user_id"
```

### 4. Create Checkpoints Container (with TTL)

```bash
az cosmosdb sql container create \
  --account-name cosmos-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --database-name agentic_post_gen \
  --name graph_checkpoints \
  --partition-key-path "/thread_id" \
  --default-ttl 5184000
```

## FAISS Index Deployment

Since FAISS index is built from local papers, you have two options:

### Option A: Pre-build and Include in Image

```dockerfile
# In Dockerfile, add:
COPY data/faiss_index ./data/faiss_index
```

Then rebuild and push the image.

### Option B: Azure Blob Storage

1. Upload FAISS index to Azure Blob Storage
2. Download at container startup
3. Add startup script to Dockerfile

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | Yes | Embeddings deployment name |
| `COSMOS_ENDPOINT` | No* | Cosmos DB endpoint (*required if CHECKPOINTER=cosmos) |
| `COSMOS_KEY` | No* | Cosmos DB key |
| `CHECKPOINTER` | No | `memory` or `cosmos` (default: memory) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key (optional) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `FAISS_INDEX_PATH` | No | Path to FAISS index (default: ./data/faiss_index) |

## Monitoring & Observability

### Application Insights

```bash
az monitor app-insights component create \
  --app app-insights-post-gen \
  --location eastus \
  --resource-group rg-agentic-post-gen \
  --application-type web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app app-insights-post-gen \
  --resource-group rg-agentic-post-gen \
  --query instrumentationKey -o tsv)

# Add to app settings
az webapp config appsettings set \
  --name app-agentic-post-gen \
  --resource-group rg-agentic-post-gen \
  --settings APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=$INSTRUMENTATION_KEY"
```

### Langfuse (Self-Hosted)

Deploy Langfuse to Azure Container Apps following their official guide, or use Langfuse Cloud.

## Security Best Practices

1. **Use Azure Key Vault** for secrets:
   ```bash
   az keyvault create \
     --name kv-post-gen \
     --resource-group rg-agentic-post-gen \
     --location eastus
   ```

2. **Enable Managed Identity** for Container Apps/App Service

3. **Configure CORS** properly in production (update `api/main.py`)

4. **Enable HTTPS only**

5. **Implement rate limiting** using Azure API Management

## CI/CD with GitHub Actions

See `.github/workflows/azure-deploy.yml` (example):

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Login to Azure
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Build and push image
        run: |
          az acr login --name acragenticpostgen
          docker build -t acragenticpostgen.azurecr.io/post-generator:${{ github.sha }} .
          docker push acragenticpostgen.azurecr.io/post-generator:${{ github.sha }}
      
      - name: Update Container App
        run: |
          az containerapp update \
            --name aca-post-generator \
            --resource-group rg-agentic-post-gen \
            --image acragenticpostgen.azurecr.io/post-generator:${{ github.sha }}
```

## Troubleshooting

### Check Logs (Container Apps)

```bash
az containerapp logs show \
  --name aca-post-generator \
  --resource-group rg-agentic-post-gen \
  --follow
```

### Check Logs (App Service)

```bash
az webapp log tail \
  --name app-agentic-post-gen \
  --resource-group rg-agentic-post-gen
```

### Common Issues

1. **FAISS index not found**: Ensure index is built and included in image or downloaded at startup
2. **Cosmos DB connection errors**: Verify endpoint and key are correct
3. **Azure OpenAI rate limits**: Implement retry logic or request quota increase
4. **Out of memory**: Increase container memory allocation

## Cost Optimization

- Use **Container Apps** consumption plan for auto-scaling
- Set appropriate **min/max replicas** based on traffic
- Use **Azure Cosmos DB serverless** for development
- Enable **Azure OpenAI rate limiting** to control costs
- Consider **spot instances** for non-production workloads

## Next Steps

1. Set up CI/CD pipeline
2. Configure custom domain
3. Implement authentication (Azure AD B2C)
4. Add rate limiting with API Management
5. Set up monitoring alerts
6. Configure backup for Cosmos DB
