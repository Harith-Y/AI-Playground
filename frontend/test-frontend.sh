#!/bin/bash

# Frontend Testing Script
# Quick automated tests to verify frontend is working

set -e

echo "🧪 ML Pipeline Frontend - Quick Test Suite"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to print test result
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        ((FAILED++))
    fi
}

# Change to frontend directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"
echo ""

# Test 1: Check if package.json exists
echo "Test 1: Checking project structure..."
if [ -f "package.json" ]; then
    test_result 0 "package.json exists"
else
    test_result 1 "package.json not found"
fi

# Test 2: Check if node_modules exists
echo ""
echo "Test 2: Checking dependencies..."
if [ -d "node_modules" ]; then
    test_result 0 "node_modules directory exists"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: node_modules not found. Run 'npm install' first."
    test_result 1 "node_modules directory missing"
fi

# Test 3: Check if src directory exists
echo ""
echo "Test 3: Checking source files..."
if [ -d "src" ]; then
    test_result 0 "src directory exists"
else
    test_result 1 "src directory not found"
fi

# Test 4: Check key files
echo ""
echo "Test 4: Checking key files..."
FILES=("src/main.tsx" "src/App.tsx" "index.html" "vite.config.ts")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        test_result 0 "$file exists"
    else
        test_result 1 "$file not found"
    fi
done

# Test 5: Try to build
echo ""
echo "Test 5: Building project..."
if npm run build > /dev/null 2>&1; then
    test_result 0 "Production build successful"
    
    # Check dist directory
    if [ -d "dist" ]; then
        test_result 0 "dist directory created"
        
        # Check dist size
        DIST_SIZE=$(du -sh dist | cut -f1)
        echo "   📦 Build size: $DIST_SIZE"
    else
        test_result 1 "dist directory not created"
    fi
else
    test_result 1 "Production build failed"
fi

# Test 6: Check Docker setup
echo ""
echo "Test 6: Checking Docker configuration..."
if [ -f "Dockerfile" ]; then
    test_result 0 "Dockerfile exists"
else
    test_result 1 "Dockerfile not found"
fi

if [ -f "nginx/nginx.conf" ]; then
    test_result 0 "nginx.conf exists"
else
    test_result 1 "nginx.conf not found"
fi

# Test 7: Check documentation
echo ""
echo "Test 7: Checking documentation..."
DOCS=("README.md" "DEPLOYMENT.md" "TESTING.md")
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        test_result 0 "$doc exists"
    else
        test_result 1 "$doc not found"
    fi
done

# Summary
echo ""
echo "=========================================="
echo "📊 Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Start dev server: npm run dev"
    echo "   2. Open http://localhost:3000"
    echo "   3. Or build Docker: docker build -t ml-pipeline-frontend ."
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Run: npm install"
    echo "   2. Check: npm run build"
    echo "   3. See: TESTING_GUIDE.md"
    exit 1
fi
