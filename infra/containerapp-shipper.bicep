// Bicep template to deploy the n8n-langfuse-shipper container app
// Reference Langfuse app template provided by user; adapted for shipper worker (no ingress)
// Best practices: parameters for dynamic values, secrets via Key Vault references, secure decorators.

@description('Name of the Container App for the shipper')
param containerapps_shipper_name string = 'n8n-langfuse-shipper'

@description('Azure region (Azure location short name) of existing Container Apps Environment, e.g. westeurope')
param location string = 'westeurope'

@description('Resource ID of existing Container Apps managed environment (same as Langfuse)')
param managedEnvironmentId string

@description('Container image to deploy (ACR fully qualified, including tag)')
param image string = 'tlteamai.azurecr.io/truelime/n8n-langfuse-shipper:latest'

@description('Workload profile name (Consumption or dedicated profile)')
param workloadProfileName string = 'Consumption'

@description('Registry server hostname (ACR)')
param registryServer string = 'tlteamai.azurecr.io'

@description('Registry username (non-secret if using dedicated pull identity)')
param registryUsername string = 'ai-kennistransport'

@secure()
@description('Key Vault secret name for ACR registry password (contains actual password)')
param kvSecretRegistryPasswordName string

@secure()
@description('Key Vault secret name containing Langfuse public key (no default)')
param kvSecretLangfusePublic string

@secure()
@description('Key Vault secret name containing Langfuse secret key (no default)')
param kvSecretLangfuseSecret string

@secure()
@description('Key Vault secret name containing Postgres password for n8n DB (no default)')
param kvSecretPostgresPassword string

@secure()
@description('Key Vault secret name containing Postgres user for n8n DB (no default)')
param kvSecretPostgresUser string

@description('Key Vault name for DB/registry secrets (se-prod-ai-kv)')
param keyVaultName string

@description('Key Vault name for Langfuse API keys (kennis vault)')
param langfuseKeyVaultName string = 'kennis'

@description('Postgres host for n8n execution database (FQDN)')
param postgresHost string = 'postgres.example.com'

@description('Postgres port for n8n database')
param postgresPort int = 5432

@description('Postgres schema containing execution tables')
param postgresSchema string = 'public'

@description('Postgres database name containing n8n tables')
param postgresDb string = 'n8n'

@description('Optional explicit DSN; if provided component vars ignored')
param pgDsn string = ''

@description('n8n table prefix (blank for none, typically n8n_)')
param dbTablePrefix string = 'n8n_'

@description('Langfuse base host (blank if using cloud default)')
param langfuseHost string = 'https://langfuse.kennistransport.com'

@description('Enable media upload feature flag')
param enableMediaUpload bool = false

@description('Maximum decoded media bytes per asset')
param mediaMaxBytes int = 25000000

@description('Extended media scan max assets per node run')
param extendedMediaScanMaxAssets int = 250

@description('Maximum number of executions processed in one run (omit or set to 0 for continuous)')
param limit int = 500

@description('Dry run processing only (no OTLP export)')
param dryRun bool = false

@description('Fetch batch size (executions per DB query)')
param fetchBatchSize int = 100

@description('Truncate field length (0 disables truncation)')
param truncateFieldLen int = 0

@description('Require execution metadata presence flag')
param requireExecutionMetadata bool = false

@description('Flush every N traces')
param flushEveryNTraces int = 1

@description('OTEL max queue size')
param otelMaxQueueSize int = 10000

@description('OTEL max export batch size')
param otelMaxExportBatchSize int = 512

@description('OTEL scheduled delay millis')
param otelScheduledDelayMillis int = 200

@description('Export queue soft limit (backpressure threshold)')
param exportQueueSoftLimit int = 5000

@description('Export sleep ms when soft limit exceeded')
param exportSleepMs int = 75

@description('Checkpoint file path (inside mounted volume)')
param checkpointFile string = '/data/.backfill_checkpoint'

@description('Log level for the application')
param logLevel string = 'INFO'

@description('CPU cores requested (integer cores; use 1 for 1 core, adjust profile if needed)')
param cpu int = 1

@description('Memory requested (Gi) whole number')
param memoryGi int = 2

@description('Minimum replicas')
param minReplicas int = 0

@description('Maximum replicas')
param maxReplicas int = 1

@description('Polling interval for autoscaler in seconds')
param pollingInterval int = 30

@description('Cooldown period for scale down in seconds')
param cooldownPeriod int = 300

@description('Target port (no ingress, internal only)')
param targetPort int = 8080

@description('Whether to enable ingress (shipper typically runs headless)')
param enableIngress bool = false

@description('Container Apps storageName (existing) for persistent checkpoint; leave blank to disable volume')
param storageName string = 'shippercheckpoint'

@description('Mount path for checkpoint volume')
param storageMountPath string = '/data'

// Derived environment variable for dry-run flag (Typer uses --dry-run / --no-dry-run)
// Runtime args are passed via container args field below (constructed inline)

// Container environment variables (non-secret values). Secrets come from Key Vault references.
var envVarsBase = [
  {
    name: 'LOG_LEVEL'
    value: logLevel
  }
  {
    name: 'DB_TABLE_PREFIX'
    value: dbTablePrefix
  }
  {
    name: 'LANGFUSE_HOST'
    value: langfuseHost
  }
  {
    name: 'ENABLE_MEDIA_UPLOAD'
    value: enableMediaUpload ? 'true' : 'false'
  }
  {
    name: 'DB_POSTGRESDB_HOST'
    value: postgresHost
  }
  {
    name: 'DB_POSTGRESDB_PORT'
    value: string(postgresPort)
  }
  {
    name: 'DB_POSTGRESDB_DATABASE'
    value: postgresDb
  }
  {
    name: 'DB_POSTGRESDB_SCHEMA'
    value: postgresSchema
  }
  {
    name: 'FETCH_BATCH_SIZE'
    value: string(fetchBatchSize)
  }
  {
    name: 'TRUNCATE_FIELD_LEN'
    value: string(truncateFieldLen)
  }
  {
    name: 'MEDIA_MAX_BYTES'
    value: string(mediaMaxBytes)
  }
  {
    name: 'EXTENDED_MEDIA_SCAN_MAX_ASSETS'
    value: string(extendedMediaScanMaxAssets)
  }
  {
    name: 'REQUIRE_EXECUTION_METADATA'
    value: requireExecutionMetadata ? 'true' : 'false'
  }
  {
    name: 'FLUSH_EVERY_N_TRACES'
    value: string(flushEveryNTraces)
  }
  {
    name: 'OTEL_MAX_QUEUE_SIZE'
    value: string(otelMaxQueueSize)
  }
  {
    name: 'OTEL_MAX_EXPORT_BATCH_SIZE'
    value: string(otelMaxExportBatchSize)
  }
  {
    name: 'OTEL_SCHEDULED_DELAY_MILLIS'
    value: string(otelScheduledDelayMillis)
  }
  {
    name: 'EXPORT_QUEUE_SOFT_LIMIT'
    value: string(exportQueueSoftLimit)
  }
  {
    name: 'EXPORT_SLEEP_MS'
    value: string(exportSleepMs)
  }
  {
    name: 'CHECKPOINT_FILE'
    value: checkpointFile
  }
]

// Conditionally append PG_DSN if provided
var envVars = pgDsn != '' ? concat(envVarsBase, [ {
  name: 'PG_DSN'
  value: pgDsn
} ]) : envVarsBase

// Secrets pulled from Key Vault via managed identity
// Langfuse secrets from kennis vault, others from se-prod-ai-kv
var secretRefs = [
  // Langfuse keys from kennis vault
  // disable-next-line no-hardcoded-env-urls
  {
    name: 'langfuse-public-key'
    keyVaultUrl: 'https://${langfuseKeyVaultName}${environment().suffixes.keyvaultDns}/secrets/${kvSecretLangfusePublic}'
    identity: 'system'
  }
  // disable-next-line no-hardcoded-env-urls
  {
    name: 'langfuse-secret-key'
    keyVaultUrl: 'https://${langfuseKeyVaultName}${environment().suffixes.keyvaultDns}/secrets/${kvSecretLangfuseSecret}'
    identity: 'system'
  }
  // DB and registry secrets from se-prod-ai-kv
  // disable-next-line no-hardcoded-env-urls
  {
    name: 'n8n-postgres-password'
    keyVaultUrl: 'https://${keyVaultName}${environment().suffixes.keyvaultDns}/secrets/${kvSecretPostgresPassword}'
    identity: 'system'
  }
  // disable-next-line no-hardcoded-env-urls
  {
    name: 'n8n-postgres-user'
    keyVaultUrl: 'https://${keyVaultName}${environment().suffixes.keyvaultDns}/secrets/${kvSecretPostgresUser}'
    identity: 'system'
  }
  // Registry password from Key Vault
  {
    // Registry password secret; name must match passwordSecretRef exactly
    name: kvSecretRegistryPasswordName
    keyVaultUrl: 'https://${keyVaultName}${environment().suffixes.keyvaultDns}/secrets/${kvSecretRegistryPasswordName}'
    identity: 'system'
  }
]

// Container env entries referencing secrets
var envSecretBindings = [
  {
    name: 'LANGFUSE_PUBLIC_KEY'
    secretRef: 'langfuse-public-key'
  }
  {
    name: 'LANGFUSE_SECRET_KEY'
    secretRef: 'langfuse-secret-key'
  }
  {
    name: 'DB_POSTGRESDB_USER'
    secretRef: 'n8n-postgres-user'
  }
  {
    name: 'DB_POSTGRESDB_PASSWORD'
    secretRef: 'n8n-postgres-password'
  }
]

// Combine plain and secret env entries
var containerEnv = concat(envVars, envSecretBindings)

resource shipper 'Microsoft.App/containerapps@2025-02-02-preview' = {
  name: containerapps_shipper_name
  location: location
  kind: 'containerapps'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
  // Only set managedEnvironmentId (environmentId field not required in this API version)
  managedEnvironmentId: managedEnvironmentId
    workloadProfileName: workloadProfileName
    configuration: {
      secrets: secretRefs
      activeRevisionsMode: 'Single'
      ingress: enableIngress ? {
        external: false
        targetPort: targetPort
        exposedPort: 0
        transport: 'Auto'
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
        allowInsecure: false
        stickySessions: {
          affinity: 'none'
        }
      } : null
      identitySettings: []
      maxInactiveRevisions: 10
      registries: [
        {
          server: registryServer
          username: registryUsername
          passwordSecretRef: kvSecretRegistryPasswordName
        }
      ]
    }
    template: {
      containers: [
        {
          name: containerapps_shipper_name
          image: image
          imageType: 'ContainerImage'
          env: containerEnv
          // Container args choose dry-run or not; limit param included
          args: dryRun ? [ '--limit', string(limit), '--dry-run' ] : [ '--limit', string(limit), '--no-dry-run' ]
          resources: {
            cpu: cpu
            memory: format('{0}Gi', memoryGi)
          }
          probes: []
          volumeMounts: storageName != '' ? [
            {
              volumeName: 'checkpoint'
              mountPath: storageMountPath
            }
          ] : []
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        cooldownPeriod: cooldownPeriod
        pollingInterval: pollingInterval
      }
  // Volume references an existing environment storage (Azure File share)
  // Create it beforehand with Azure CLI (see deployment instructions)
  volumes: storageName != '' ? [
        {
          name: 'checkpoint'
          storageName: storageName
        }
      ] : []
    }
  }
}

// Output key resource identifiers for downstream scripting
@description('Container App Resource ID')
output shipperResourceId string = shipper.id

@description('Latest revision name')
output revisionName string = shipper.properties.latestRevisionName
