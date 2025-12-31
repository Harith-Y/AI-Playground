# Frontend Testing Script (PowerShell)
# Quick automated tests to verify frontend is working

Write-Host "🧪 ML Pipeline Frontend - Quick Test Suite" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$PASSED = 0
$FAILED = 0

function Test-Result {
    param($Success, $Message)
    if ($Success) {
        Write-Host "✓ PASS: $Message" -ForegroundColor Green
        $script:PASSED++
    } else {
        Write-Host "✗ FAIL: $Message" -ForegroundColor Red
        $script:FAILED++
    }
}

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "📁 Current directory: $(Get-Location)"
Write-Host ""

# Test 1: Check if package.json exists
Write-Host "Test 1: Checking project structure..."
Test-Result (Test-Path "package.json") "package.json exists"

# Test 2: Check if node_modules exists
Write-Host ""
Write-Host "Test 2: Checking dependencies..."
if (Test-Path "node_modules") {
    Test-Result $true "node_modules directory exists"
} else {
    Write-Host "⚠ WARNING: node_modules not found. Run 'npm install' first." -ForegroundColor Yellow
    Test-Result $false "node_modules directory missing"
}

# Test 3: Check if src directory exists
Write-Host ""
Write-Host "Test 3: Checking source files..."
Test-Result (Test-Path "src") "src directory exists"

# Test 4: Check key files
Write-Host ""
Write-Host "Test 4: Checking key files..."
$files = @("src/main.tsx", "src/App.tsx", "index.html", "vite.config.ts")
foreach ($file in $files) {
    Test-Result (Test-Path $file) "$file exists"
}

# Test 5: Try to build
Write-Host ""
Write-Host "Test 5: Building project..."
try {
    $buildOutput = npm run build 2>&1
    if ($LASTEXITCODE -eq 0) {
        Test-Result $true "Production build successful"
        
        if (Test-Path "dist") {
            Test-Result $true "dist directory created"
            
            $distSize = (Get-ChildItem -Path "dist" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
            Write-Host "   📦 Build size: $([math]::Round($distSize, 2)) MB"
        } else {
            Test-Result $false "dist directory not created"
        }
    } else {
        Test-Result $false "Production build failed"
    }
} catch {
    Test-Result $false "Production build failed with error"
}

# Test 6: Check Docker setup
Write-Host ""
Write-Host "Test 6: Checking Docker configuration..."
Test-Result (Test-Path "Dockerfile") "Dockerfile exists"
Test-Result (Test-Path "nginx/nginx.conf") "nginx.conf exists"

# Test 7: Check documentation
Write-Host ""
Write-Host "Test 7: Checking documentation..."
$docs = @("README.md", "DEPLOYMENT.md", "TESTING.md")
foreach ($doc in $docs) {
    Test-Result (Test-Path $doc) "$doc exists"
}

# Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "📊 Test Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Passed: $PASSED" -ForegroundColor Green
Write-Host "Failed: $FAILED" -ForegroundColor Red
Write-Host ""

if ($FAILED -eq 0) {
    Write-Host "✓ All tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Next steps:"
    Write-Host "   1. Start dev server: npm run dev"
    Write-Host "   2. Open http://localhost:3000"
    Write-Host "   3. Or build Docker: docker build -t ml-pipeline-frontend ."
    exit 0
} else {
    Write-Host "✗ Some tests failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Troubleshooting:"
    Write-Host "   1. Run: npm install"
    Write-Host "   2. Check: npm run build"
    Write-Host "   3. See: TESTING_GUIDE.md"
    exit 1
}
