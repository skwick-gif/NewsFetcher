param(
    [switch]$SkipBuild,
    [Alias("Host")][string]$IbkrHost,
    [int]$Port,
    [int]$ClientId
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bridgeDir = Join-Path $repoRoot "IBKR\csharp_bridge"
if (-not (Test-Path $bridgeDir)) {
    Write-Error "IBKR bridge directory not found at $bridgeDir"
    exit 1
}

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Error "dotnet CLI not found. Install the .NET SDK to run the bridge."
    exit 1
}

if (-not $SkipBuild) {
    Push-Location $bridgeDir
    Write-Host "Restoring IBKR bridge dependencies..." -ForegroundColor Cyan
    dotnet restore --nologo
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "dotnet restore failed"
        exit 1
    }
    Write-Host "Building IBKR bridge..." -ForegroundColor Cyan
    dotnet build --configuration Release --nologo
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "dotnet build failed"
        exit 1
    }
    Pop-Location
}

$commandLines = @()
$commandLines += ('Set-Location "{0}"' -f $bridgeDir)
$commandLines += "`$env:ASPNETCORE_ENVIRONMENT = 'Development'"
if ($IbkrHost) { $commandLines += "`$env:IBKR__Host = '$IbkrHost'" }
if ($Port) { $commandLines += "`$env:IBKR__Port = '$Port'" }
if ($ClientId) { $commandLines += "`$env:IBKR__ClientId = '$ClientId'" }
$commandLines += "dotnet run --configuration Release --no-launch-profile --urls http://localhost:5080"
$fullCommand = $commandLines -join '; '

Write-Host "Starting IBKR bridge on http://localhost:5080 ..." -ForegroundColor Green
$bridgeProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", $fullCommand -PassThru

Write-Host "Bridge process (PID $($bridgeProcess.Id)) launched in a new window." -ForegroundColor Yellow
Write-Host "Waiting for the bridge health endpoint..." -ForegroundColor Yellow

$healthUrl = "http://localhost:5080/health"
$maxAttempts = 20
for ($i = 0; $i -lt $maxAttempts; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "IBKR bridge is online at $healthUrl" -ForegroundColor Green
            break
        }
    } catch {
        if ($i -eq $maxAttempts - 1) {
            Write-Warning "Could not verify bridge health. Check the bridge window for details."
        }
    }
}

Write-Host "Keep the new PowerShell window open while using the dashboard." -ForegroundColor Cyan
