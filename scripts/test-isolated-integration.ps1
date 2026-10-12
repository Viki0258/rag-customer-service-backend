# Runs only against newly created, disposable services, never inherited database URLs.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$composeFile = Join-Path $repoRoot 'compose.integration.yaml'
$projectName = 'rag-integration-' + [guid]::NewGuid().ToString('N')
$envNames = @('WHATSAPP_SUPPORT_DATABASE_URL', 'WHATSAPP_SUPPORT_REDIS_URL',
    'WHATSAPP_SUPPORT_RAG_MODE', 'PYTHONPATH')
$previous = @{}
foreach ($name in $envNames) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$testExit = 1
$created = $false
Push-Location $repoRoot
try {
    & docker info --format '{{.ServerVersion}}'
    if ($LASTEXITCODE -ne 0) { throw 'Docker engine must be running.' }
    & py -3.12 -c 'import asyncpg, redis, pytest, sqlalchemy'
    if ($LASTEXITCODE -ne 0) { throw 'Install project runtime and test extras first.' }
    # Cleanup is scoped to our newly generated project, including partially started services.
    $created = $true
    & docker compose -f $composeFile -p $projectName up -d --wait --wait-timeout 60
    if ($LASTEXITCODE -ne 0) { throw 'Disposable integration services failed to start.' }
    $pgEndpoint = (& docker compose -f $composeFile -p $projectName port postgres 5432)
    if ($LASTEXITCODE -ne 0 -or $pgEndpoint -notmatch '^127\.0\.0\.1:(\d+)$') {
        throw 'Unexpected PostgreSQL test endpoint.'
    }
    $pgPort = $Matches[1]
    $redisEndpoint = (& docker compose -f $composeFile -p $projectName port redis 6379)
    if ($LASTEXITCODE -ne 0 -or $redisEndpoint -notmatch '^127\.0\.0\.1:(\d+)$') {
        throw 'Unexpected Redis test endpoint.'
    }
    $redisPort = $Matches[1]
    $env:WHATSAPP_SUPPORT_DATABASE_URL = "postgresql+asyncpg://integration_test:local_test_only@127.0.0.1:${pgPort}/whatsapp_support_test"
    $env:WHATSAPP_SUPPORT_REDIS_URL = "redis://127.0.0.1:${redisPort}/0"
    $env:WHATSAPP_SUPPORT_RAG_MODE = 'fake'
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    & py -3.12 -m pytest tests/integration -q -rs
    $testExit = $LASTEXITCODE
} finally {
    if ($created) {
        & docker compose -f $composeFile -p $projectName down --timeout 10
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Cleanup failed for disposable project $projectName; inspect it manually."
            $testExit = 1
        }
    }
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
    Pop-Location
}
exit $testExit
