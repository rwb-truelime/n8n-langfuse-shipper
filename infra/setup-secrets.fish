#!/usr/bin/env fish
# Setup script to create missing Langfuse API key secrets in Azure Key Vault
# Run this ONCE before deploying the container app

set KV_NAME "se-prod-ai-kv"

echo "============================================"
echo "Langfuse Shipper - Secret Setup"
echo "============================================"
echo ""
echo "This script will help you create the required Langfuse API key secrets."
echo ""
echo "PREREQUISITES:"
echo "1. You must have created a Langfuse API key pair in your Langfuse UI:"
echo "   - Navigate to: https://langfuse.kennistransport.com/settings"
echo "   - Go to 'API Keys' section"
echo "   - Click 'Create new API key'"
echo "   - Copy both PUBLIC KEY and SECRET KEY"
echo ""
echo "2. You must have Key Vault access (az keyvault set-policy if needed)"
echo ""
echo "============================================"
echo ""

# Check if user has access to Key Vault
if not az keyvault show --name $KV_NAME --query id -o tsv &>/dev/null
    echo "ERROR: Cannot access Key Vault '$KV_NAME'"
    echo "Run: az keyvault set-policy --name $KV_NAME --upn (az account show --query user.name -o tsv) --secret-permissions get list set delete"
    exit 1
end

echo "✓ Key Vault access confirmed"
echo ""

# Check which secrets already exist
set EXISTING_SECRETS (az keyvault secret list --vault-name $KV_NAME --query "[].name" -o tsv)

function secret_exists
    echo $EXISTING_SECRETS | grep -q "^$argv[1]\$"
end

# Langfuse Public Key
if secret_exists prd-langfuse-public-key
    echo "✓ Secret 'prd-langfuse-public-key' already exists"
else
    echo "Creating 'prd-langfuse-public-key'..."
    read -P "Enter Langfuse PUBLIC KEY (starts with 'pk-lf-'): " -s PUBLIC_KEY
    echo ""

    if test -z "$PUBLIC_KEY"
        echo "ERROR: Public key cannot be empty"
        exit 1
    end

    az keyvault secret set --vault-name $KV_NAME --name prd-langfuse-public-key --value "$PUBLIC_KEY" --output none
    echo "✓ Created 'prd-langfuse-public-key'"
end

echo ""

# Langfuse Secret Key
if secret_exists prd-langfuse-secret-key
    echo "✓ Secret 'prd-langfuse-secret-key' already exists"
else
    echo "Creating 'prd-langfuse-secret-key'..."
    read -P "Enter Langfuse SECRET KEY (starts with 'sk-lf-'): " -s SECRET_KEY
    echo ""

    if test -z "$SECRET_KEY"
        echo "ERROR: Secret key cannot be empty"
        exit 1
    end

    az keyvault secret set --vault-name $KV_NAME --name prd-langfuse-secret-key --value "$SECRET_KEY" --output none
    echo "✓ Created 'prd-langfuse-secret-key'"
end

echo ""
echo "============================================"
echo "✓ All secrets configured!"
echo "============================================"
echo ""
echo "Existing secrets mapped:"
echo "  - Registry password: truelime-tlteamai-azurecr-io (already exists)"
echo "  - Postgres user: prd-n8n-postgresdb-user (already exists)"
echo "  - Postgres password: prd-n8n-postgresdb-password (already exists)"
echo ""
echo "You can now deploy with:"
echo "  set -x RG_NAME rg-web"
echo "  az deployment group create \\"
echo "    --resource-group \$RG_NAME \\"
echo "    --name containerapp-shipper \\"
echo "    --template-file infra/containerapp-shipper.bicep \\"
echo "    --parameters @infra/shipper.parameters.json"
echo ""
