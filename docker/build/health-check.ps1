# Docker Build Health Check
# Run this script to verify Docker build configuration

$ErrorActionPreference = "Stop"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Docker Build Health Check" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$checks = @()
$passed = 0
$failed = 0

# Check Docker installation
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "  ✓ Docker installed: $dockerVersion" -ForegroundColor Green
    $checks += @{Name = "Docker Installation"; Status = "PASS"; Details = $dockerVersion}
    $passed++
}
catch {
    Write-Host "  ✗ Docker not found" -ForegroundColor Red
    $checks += @{Name = "Docker Installation"; Status = "FAIL"; Details = "Docker not installed"}
    $failed++
}

# Check Docker daemon
Write-Host "Checking Docker daemon..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  ✓ Docker daemon running" -ForegroundColor Green
    $checks += @{Name = "Docker Daemon"; Status = "PASS"; Details = "Running"}
    $passed++
}
catch {
    Write-Host "  ✗ Docker daemon not running" -ForegroundColor Red
    $checks += @{Name = "Docker Daemon"; Status = "FAIL"; Details = "Not running"}
    $failed++
}

# Check Docker Buildx
Write-Host "Checking Docker Buildx..." -ForegroundColor Yellow
try {
    $buildxVersion = docker buildx version
    Write-Host "  ✓ Docker Buildx available: $buildxVersion" -ForegroundColor Green
    $checks += @{Name = "Docker Buildx"; Status = "PASS"; Details = $buildxVersion}
    $passed++
}
catch {
    Write-Host "  ⚠ Docker Buildx not available (multi-platform builds won't work)" -ForegroundColor Yellow
    $checks += @{Name = "Docker Buildx"; Status = "WARN"; Details = "Not available"}
}

# Check Dockerfiles
Write-Host "Checking Dockerfiles..." -ForegroundColor Yellow

$dockerfiles = @(
    "backend\Dockerfile",
    "backend\Dockerfile.dev",
    "frontend\Dockerfile"
)

foreach ($dockerfile in $dockerfiles) {
    if (Test-Path $dockerfile) {
        Write-Host "  ✓ Found $dockerfile" -ForegroundColor Green
        $checks += @{Name = "Dockerfile: $dockerfile"; Status = "PASS"; Details = "Exists"}
        $passed++
    }
    else {
        Write-Host "  ✗ Missing $dockerfile" -ForegroundColor Red
        $checks += @{Name = "Dockerfile: $dockerfile"; Status = "FAIL"; Details = "Not found"}
        $failed++
    }
}

# Check .dockerignore files
Write-Host "Checking .dockerignore files..." -ForegroundColor Yellow

$dockerignores = @(
    "backend\.dockerignore",
    "frontend\.dockerignore"
)

foreach ($dockerignore in $dockerignores) {
    if (Test-Path $dockerignore) {
        Write-Host "  ✓ Found $dockerignore" -ForegroundColor Green
        $checks += @{Name = ".dockerignore: $dockerignore"; Status = "PASS"; Details = "Exists"}
        $passed++
    }
    else {
        Write-Host "  ⚠ Missing $dockerignore (builds may be slower)" -ForegroundColor Yellow
        $checks += @{Name = ".dockerignore: $dockerignore"; Status = "WARN"; Details = "Not found"}
    }
}

# Check docker-compose files
Write-Host "Checking docker-compose files..." -ForegroundColor Yellow

$composeFiles = @(
    "docker-compose.yml",
    "docker-compose.dev.yml",
    "docker-compose.production.yml"
)

foreach ($file in $composeFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ Found $file" -ForegroundColor Green
        # Validate docker-compose file
        try {
            docker-compose -f $file config | Out-Null
            Write-Host "    ✓ Valid configuration" -ForegroundColor Green
            $checks += @{Name = "Compose file: $file"; Status = "PASS"; Details = "Valid"}
            $passed++
        }
        catch {
            Write-Host "    ✗ Invalid configuration" -ForegroundColor Red
            $checks += @{Name = "Compose file: $file"; Status = "FAIL"; Details = "Invalid"}
            $failed++
        }
    }
    else {
        Write-Host "  ✗ Missing $file" -ForegroundColor Red
        $checks += @{Name = "Compose file: $file"; Status = "FAIL"; Details = "Not found"}
        $failed++
    }
}

# Check build scripts
Write-Host "Checking build scripts..." -ForegroundColor Yellow

$buildScripts = @(
    "docker\build\docker-build.sh",
    "docker\build\docker-build.ps1"
)

foreach ($script in $buildScripts) {
    if (Test-Path $script) {
        Write-Host "  ✓ Found $script" -ForegroundColor Green
        $checks += @{Name = "Build script: $script"; Status = "PASS"; Details = "Exists"}
        $passed++
    }
    else {
        Write-Host "  ✗ Missing $script" -ForegroundColor Red
        $checks += @{Name = "Build script: $script"; Status = "FAIL"; Details = "Not found"}
        $failed++
    }
}

# Check environment files
Write-Host "Checking environment files..." -ForegroundColor Yellow

$envFiles = @(
    ".env.docker.example"
)

foreach ($file in $envFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ Found $file" -ForegroundColor Green
        $checks += @{Name = "Env file: $file"; Status = "PASS"; Details = "Exists"}
        $passed++
    }
    else {
        Write-Host "  ⚠ Missing $file (example config not available)" -ForegroundColor Yellow
        $checks += @{Name = "Env file: $file"; Status = "WARN"; Details = "Not found"}
    }
}

# Check disk space
Write-Host "Checking disk space..." -ForegroundColor Yellow
$drive = Get-PSDrive -Name C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)

if ($freeSpaceGB -gt 10) {
    Write-Host "  ✓ Sufficient disk space: ${freeSpaceGB}GB free" -ForegroundColor Green
    $checks += @{Name = "Disk Space"; Status = "PASS"; Details = "${freeSpaceGB}GB free"}
    $passed++
}
elseif ($freeSpaceGB -gt 5) {
    Write-Host "  ⚠ Low disk space: ${freeSpaceGB}GB free" -ForegroundColor Yellow
    $checks += @{Name = "Disk Space"; Status = "WARN"; Details = "${freeSpaceGB}GB free"}
}
else {
    Write-Host "  ✗ Insufficient disk space: ${freeSpaceGB}GB free" -ForegroundColor Red
    $checks += @{Name = "Disk Space"; Status = "FAIL"; Details = "Only ${freeSpaceGB}GB free"}
    $failed++
}

# Summary
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Health Check Summary" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor Red
Write-Host ""

if ($failed -eq 0) {
    Write-Host "✓ All checks passed! Docker build system is ready." -ForegroundColor Green
    exit 0
}
else {
    Write-Host "✗ Some checks failed. Please fix the issues above." -ForegroundColor Red
    exit 1
}
