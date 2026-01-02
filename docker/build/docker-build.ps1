# ====================================
# Docker Build Script for AI-Playground (PowerShell)
# ====================================
#
# Usage:
#   .\docker\build\docker-build.ps1 [OPTIONS]
#
# Options:
#   -Service <name>      Build specific service (backend, frontend, all) [default: all]
#   -Env <environment>   Environment (dev, prod) [default: prod]
#   -Platform <arch>     Platform (linux/amd64, linux/arm64) [default: linux/amd64]
#   -Push                Push to registry after build
#   -NoCache             Build without using cache
#   -Tag <version>       Custom tag version [default: latest]
#   -Registry <url>      Docker registry URL [default: docker.io]
#   -Help                Show this help message
# ====================================

param(
    [string]$Service = "all",
    [string]$Env = "prod",
    [string]$Platform = "linux/amd64",
    [switch]$Push = $false,
    [switch]$NoCache = $false,
    [string]$Tag = "latest",
    [string]$Registry = "docker.io",
    [switch]$Help = $false
)

# Colors
$colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Cyan"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# Show help
if ($Help) {
    Get-Content $PSCommandPath | Select-String "^#" | ForEach-Object { $_.Line.Substring(2) }
    exit 0
}

# Validate inputs
if ($Env -notin @("dev", "prod")) {
    Write-ColorOutput "Error: Environment must be 'dev' or 'prod'" $colors.Red
    exit 1
}

if ($Service -notin @("backend", "frontend", "all")) {
    Write-ColorOutput "Error: Service must be 'backend', 'frontend', or 'all'" $colors.Red
    exit 1
}

if ($Platform -notin @("linux/amd64", "linux/arm64")) {
    Write-ColorOutput "Error: Platform must be 'linux/amd64' or 'linux/arm64'" $colors.Red
    exit 1
}

# Get project root
$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$ProjectName = "ai-playground"

# Display configuration
Write-ColorOutput "=====================================" $colors.Blue
Write-ColorOutput "AI-Playground Docker Build" $colors.Blue
Write-ColorOutput "=====================================" $colors.Blue
Write-ColorOutput "Configuration:" $colors.Green
Write-ColorOutput "  Service: $Service" $colors.Yellow
Write-ColorOutput "  Environment: $Env" $colors.Yellow
Write-ColorOutput "  Platform: $Platform" $colors.Yellow
Write-ColorOutput "  Tag: $Tag" $colors.Yellow
Write-ColorOutput "  Registry: $Registry" $colors.Yellow
Write-ColorOutput "  Push: $Push" $colors.Yellow
Write-ColorOutput "=====================================" $colors.Blue
Write-Host ""

# Function to build backend
function Build-Backend {
    Write-ColorOutput "Building Backend Service..." $colors.Green
    
    $Dockerfile = if ($Env -eq "dev") { "Dockerfile.dev" } else { "Dockerfile" }
    $Target = if ($Env -eq "prod") { "production" } else { "" }
    $RequirementsFile = if ($Env -eq "prod") { "requirements.render.txt" } else { "requirements.txt" }
    $ImageName = "$Registry/${ProjectName}-backend:$Tag"
    
    Write-ColorOutput "Image: $ImageName" $colors.Yellow
    Write-ColorOutput "Dockerfile: backend/$Dockerfile" $colors.Yellow
    
    Push-Location "$ProjectRoot\backend"
    
    try {
        $buildArgs = @(
            "build",
            "--platform", $Platform,
            "--build-arg", "REQUIREMENTS_FILE=$RequirementsFile",
            "-t", $ImageName,
            "-f", $Dockerfile
        )
        
        if ($Target) {
            $buildArgs += "--target", $Target
        }
        
        if ($NoCache) {
            $buildArgs += "--no-cache"
        }
        
        $buildArgs += "."
        
        & docker $buildArgs
        
        if ($LASTEXITCODE -ne 0) {
            throw "Backend build failed"
        }
        
        if ($Push) {
            Write-ColorOutput "Pushing backend image..." $colors.Green
            docker push $ImageName
            
            if ($LASTEXITCODE -ne 0) {
                throw "Backend push failed"
            }
        }
        
        Write-ColorOutput "✓ Backend build complete" $colors.Green
        Write-Host ""
    }
    finally {
        Pop-Location
    }
}

# Function to build frontend
function Build-Frontend {
    Write-ColorOutput "Building Frontend Service..." $colors.Green
    
    $Target = if ($Env -eq "prod") { "production" } else { "development" }
    $ImageName = "$Registry/${ProjectName}-frontend:$Tag"
    
    Write-ColorOutput "Image: $ImageName" $colors.Yellow
    Write-ColorOutput "Target: $Target" $colors.Yellow
    
    Push-Location "$ProjectRoot\frontend"
    
    try {
        $buildArgs = @(
            "build",
            "--platform", $Platform,
            "--target", $Target,
            "-t", $ImageName,
            "-f", "Dockerfile"
        )
        
        # Add build args if .env.production exists
        if (Test-Path ".env.production") {
            $viteApiUrl = $env:VITE_API_URL ?? "http://localhost:8000"
            $viteWsUrl = $env:VITE_WS_URL ?? "ws://localhost:8000/ws"
            
            $buildArgs += "--build-arg", "VITE_API_URL=$viteApiUrl"
            $buildArgs += "--build-arg", "VITE_WS_URL=$viteWsUrl"
        }
        
        if ($NoCache) {
            $buildArgs += "--no-cache"
        }
        
        $buildArgs += "."
        
        & docker $buildArgs
        
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed"
        }
        
        if ($Push) {
            Write-ColorOutput "Pushing frontend image..." $colors.Green
            docker push $ImageName
            
            if ($LASTEXITCODE -ne 0) {
                throw "Frontend push failed"
            }
        }
        
        Write-ColorOutput "✓ Frontend build complete" $colors.Green
        Write-Host ""
    }
    finally {
        Pop-Location
    }
}

# Main build logic
try {
    Set-Location $ProjectRoot
    
    if ($Service -eq "backend" -or $Service -eq "all") {
        Build-Backend
    }
    
    if ($Service -eq "frontend" -or $Service -eq "all") {
        Build-Frontend
    }
    
    Write-ColorOutput "=====================================" $colors.Blue
    Write-ColorOutput "✓ Build Complete!" $colors.Green
    Write-ColorOutput "=====================================" $colors.Blue
    
    # Print next steps
    Write-Host ""
    Write-ColorOutput "Next steps:" $colors.Yellow
    if (-not $Push) {
        Write-ColorOutput "  1. Test locally: docker-compose up" $colors.Blue
        Write-ColorOutput "  2. Push to registry: .\docker\build\docker-build.ps1 -Service $Service -Push" $colors.Blue
    }
    else {
        Write-ColorOutput "  1. Pull on server: docker pull $Registry/${ProjectName}-${Service}:$Tag" $colors.Blue
        Write-ColorOutput "  2. Deploy: docker-compose up -d" $colors.Blue
    }
    
    exit 0
}
catch {
    Write-ColorOutput "Error: $_" $colors.Red
    exit 1
}
