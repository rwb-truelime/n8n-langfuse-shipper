using './containerapp-shipper.bicep'

// Example parameters file for deploying n8n-langfuse-shipper Container App.
// Adjust subscription / resource IDs and secret names for your environment.
// Do NOT commit real secret values; only secret names are listed.
// All secret values are resolved via Key Vault + managed identity at runtime.

param managedEnvironmentId = '/subscriptions/<subscription-id>/resourceGroups/<rg-name>/providers/Microsoft.App/managedEnvironments/<env-name>'
param image = 'tlteamai.azurecr.io/truelime/n8n-langfuse-shipper:latest'
param workloadProfileName = 'Consumption'
param registryServer = 'tlteamai.azurecr.io'
param registryUsername = 'ai-kennistransport'
param kvSecretRegistryPasswordName = 'truelime-tlteamai-azurecr-io'
param keyVaultName = 'se-prod-ai-kv'
param kvSecretLangfusePublic = 'langfuse-public-key'
param kvSecretLangfuseSecret = 'langfuse-secret-key'
param kvSecretPostgresUser = 'n8n-postgres-user'
param kvSecretPostgresPassword = 'n8n-postgres-password'
param postgresHost = 'kennis.postgres.database.azure.com'
param postgresPort = 5432
param postgresDb = 'n8n'
param postgresSchema = 'public'
param pgDsn = '' // Leave empty to build from components above
param dbTablePrefix = 'n8n_'
param langfuseHost = 'https://langfuse.kennistransport.com'
param enableMediaUpload = true
param fetchBatchSize = 100
param truncateFieldLen = 0
param mediaMaxBytes = 25000000
param extendedMediaScanMaxAssets = 250
param flushEveryNTraces = 1
param otelMaxQueueSize = 10000
param otelMaxExportBatchSize = 512
param otelScheduledDelayMillis = 200
param exportQueueSoftLimit = 5000
param exportSleepMs = 75
param requireExecutionMetadata = false
param logLevel = 'INFO'
param cpu = 1
param memoryGi = 1
param minReplicas = 0
param maxReplicas = 1
param pollingInterval = 30
param cooldownPeriod = 300
param targetPort = 8080
param enableIngress = false
param storageName = 'shippercheckpoint' // Existing env storage name
param storageMountPath = '/data'
param checkpointFile = '/data/.backfill_checkpoint'
param limit = 500
param dryRun = false
