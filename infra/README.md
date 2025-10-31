
### Azure Deployment (Bicep)

Infrastructure as Code for deploying the Container App is provided in
`infra/containerapp-shipper.bicep`.

#### Parameters Overview (Full)
| Param | Purpose |
|-------|---------|
| `containerapps_shipper_name` | Container App name. |
| `location` | Azure region. |
| `managedEnvironmentId` | Container Apps Environment resource ID. |
| `image` | ACR image reference. |
| `workloadProfileName` | Workload profile (Consumption etc.). |
| `registryServer` | ACR server hostname. |
| `registryUsername` | ACR username. |
| `kvSecretRegistryPasswordName` | Key Vault secret name for ACR password. |
| `keyVaultName` | Key Vault name hosting secrets. |
| `kvSecretLangfusePublic` / `kvSecretLangfuseSecret` | Secret names for Langfuse keys. |
| `kvSecretPostgresUser` / `kvSecretPostgresPassword` | Secret names for Postgres creds. |
| `postgresHost` / `postgresPort` / `postgresDb` / `postgresSchema` | Postgres components. |
| `pgDsn` | Full DSN override (precedence over component vars). |
| `dbTablePrefix` | n8n table prefix (blank allowed). |
| `langfuseHost` | Optional Langfuse base host. |
| `enableMediaUpload` | Enable media token upload. |
| `fetchBatchSize` | Executions per DB batch. |
| `truncateFieldLen` | Truncation length (0 disables). |
| `mediaMaxBytes` | Max decoded bytes per asset. |
| `extendedMediaScanMaxAssets` | Binary discovery cap. |
| `flushEveryNTraces` | Forced flush cadence. |
| `otelMaxQueueSize` | Span processor queue size. |
| `otelMaxExportBatchSize` | Max spans per export request. |
| `otelScheduledDelayMillis` | Max batch delay (ms). |
| `exportQueueSoftLimit` | Backpressure threshold. |
| `exportSleepMs` | Sleep duration (ms) on backpressure. |
| `requireExecutionMetadata` | Metadata presence filter. |
| `logLevel` | Log verbosity. |
| `cpu` / `memoryGi` | Resource requests. |
| `minReplicas` / `maxReplicas` | Replica bounds. |
| `pollingInterval` / `cooldownPeriod` | Autoscale timings. |
| `targetPort` / `enableIngress` | Ingress configuration. |
| `storageName` / `storageMountPath` | Azure File share volume config. |
| `checkpointFile` | Persistent checkpoint path. |
| `limit` | Execution cap (0 continuous). |
| `dryRun` | Mapping only; no export. |

Notes: secret parameters are names (Key Vault resolves values). Use a single replica unless you implement id range partitioning to avoid redundant work.

#### Example `.bicepparam` (Optional)
Create `infra/shipper.bicepparam` locally (not committed if you store secrets):
```bicep
using 'containerapp-shipper.bicep'

param managedEnvironmentId = '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.App/managedEnvironments/<env>'
param image = 'tlteamai.azurecr.io/truelime/n8n-langfuse-shipper:latest'
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
param dbTablePrefix = 'n8n_'
param enableMediaUpload = false
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
param limit = 500
param dryRun = false
param storageName = 'shippercheckpoint'
param checkpointFile = '/data/.backfill_checkpoint'
```

Deploy with params file:
```fish
az deployment group create \
	--resource-group $RG_NAME \
	--template-file infra/containerapp-shipper.bicep \
	--parameters @infra/shipper.bicepparam
```

#### Provision Azure File Share Storage
```fish
set -x RG_NAME <resource-group>
set -x STORAGE_ACCOUNT kennis
set -x FILE_SHARE langfuse-shipper-persistence
set -x ENV_NAME <container-app-env>
set -x STORAGE_KEY (az storage account keys list --resource-group $RG_NAME --account-name $STORAGE_ACCOUNT --query '[0].value' -o tsv)
az storage share-rm create --resource-group $RG_NAME --storage-account $STORAGE_ACCOUNT --name $FILE_SHARE --quota 100 --account-key $STORAGE_KEY
az containerapp env storage set --name $ENV_NAME --resource-group $RG_NAME --storage-name shippercheckpoint --access-mode ReadWrite --account-name $STORAGE_ACCOUNT --share-name $FILE_SHARE --account-key $STORAGE_KEY
```

#### Deploy / Update
```fish
az deployment group create \
	--resource-group $RG_NAME \
	--template-file infra/containerapp-shipper.bicep \
	--parameters managedEnvironmentId=$ENV_ID image=$IMAGE kvSecretRegistryPasswordName=truelime-tlteamai-azurecr-io \
		keyVaultName=se-prod-ai-kv kvSecretLangfusePublic=langfuse-public-key kvSecretLangfuseSecret=langfuse-secret-key \
		kvSecretPostgresUser=n8n-postgres-user kvSecretPostgresPassword=n8n-postgres-password postgresHost=kennis.postgres.database.azure.com \
		postgresPort=5432 postgresDb=n8n postgresSchema=public dbTablePrefix=n8n_ enableMediaUpload=false fetchBatchSize=100 truncateFieldLen=0 \
		mediaMaxBytes=25000000 extendedMediaScanMaxAssets=250 flushEveryNTraces=1 otelMaxQueueSize=10000 otelMaxExportBatchSize=512 \
		otelScheduledDelayMillis=200 exportQueueSoftLimit=5000 exportSleepMs=75 limit=500 dryRun=false storageName=shippercheckpoint
```

#### Scaling Guidance
- Increase `fetchBatchSize` cautiously; watch CPU & memory.
- Raise `otelMaxQueueSize` only if exporter backpressure metadata shows frequent sleeps.
- Horizontal scaling (replicas >1) risks duplicate processing (deterministic IDs prevent logical duplication in Langfuse but extra load); prefer single replica unless partitioning by id range.

#### Secret Management Notes
- All secret names passed as parameters are Key Vault secret names (NOT values). Values are resolved by Container Apps runtime via managed identity.
- Rotate credentials by updating the Key Vault secret; redeploy only if environment settings (not secret values) change.

#### Checkpoint Persistence
- Ensure `storageName` points to a registered storage resource; volume mounts at `/data` and persists `.backfill_checkpoint`.
- Remove `storageName` to run stateless (not recommended for large backfills).

#### Troubleshooting
| Symptom | Action |
|---------|--------|
| Pod restarts lose progress | Verify volume mount (`az containerapp show ... --query properties.template.containers[0].volumeMounts`). |
| No media tokens | Set `enableMediaUpload=true` and confirm outbound network. |
| High latency exporting | Tune `otelScheduledDelayMillis` and `flushEveryNTraces`. |
| Backpressure sleeps frequent | Increase `otelMaxQueueSize` or lower `fetchBatchSize`. |
| Empty execution processing | Check `dbTablePrefix`, `postgresSchema`, DSN correctness. |
| Missing env var | Confirm parameter passed; inspect with `az containerapp show`. |
| Media tokens absent | Set `enableMediaUpload=true`; verify egress & size vs `mediaMaxBytes`. |
| Unexpected truncation | Ensure `truncateFieldLen=0`. |

---
