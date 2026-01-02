#!/bin/bash
# ====================================
# Docker Build Script for AI-Playground
# ====================================
#
# Usage:
#   ./docker/build/docker-build.sh [OPTIONS]
#
# Options:
#   --service <name>     Build specific service (backend, frontend, all)
#   --env <environment>  Environment (dev, prod) [default: prod]
#   --platform <arch>    Platform (linux/amd64, linux/arm64, both) [default: linux/amd64]
#   --push               Push to registry after build
#   --no-cache           Build without using cache
#   --tag <version>      Custom tag version [default: latest]
#   --registry <url>     Docker registry URL [default: docker.io]
#   --help               Show this help message
# ====================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SERVICE="all"
ENVIRONMENT="prod"
PLATFORM="linux/amd64"
PUSH=false
NO_CACHE=""
TAG="latest"
REGISTRY="docker.io"
PROJECT_NAME="ai-playground"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --service)
            SERVICE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --help)
            head -n 20 "$0" | grep "^#" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    echo -e "${RED}Error: Environment must be 'dev' or 'prod'${NC}"
    exit 1
fi

if [[ "$SERVICE" != "backend" && "$SERVICE" != "frontend" && "$SERVICE" != "all" ]]; then
    echo -e "${RED}Error: Service must be 'backend', 'frontend', or 'all'${NC}"
    exit 1
fi

# Set up platform flags
PLATFORM_FLAG=""
if [[ "$PLATFORM" == "both" ]]; then
    PLATFORM_FLAG="--platform linux/amd64,linux/arm64"
    # Multi-platform builds require buildx
    if ! docker buildx version &> /dev/null; then
        echo -e "${RED}Error: docker buildx is required for multi-platform builds${NC}"
        exit 1
    fi
elif [[ "$PLATFORM" != "linux/amd64" && "$PLATFORM" != "linux/arm64" ]]; then
    echo -e "${RED}Error: Platform must be 'linux/amd64', 'linux/arm64', or 'both'${NC}"
    exit 1
else
    PLATFORM_FLAG="--platform $PLATFORM"
fi

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}AI-Playground Docker Build${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Service: ${YELLOW}$SERVICE${NC}"
echo -e "  Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "  Platform: ${YELLOW}$PLATFORM${NC}"
echo -e "  Tag: ${YELLOW}$TAG${NC}"
echo -e "  Registry: ${YELLOW}$REGISTRY${NC}"
echo -e "  Push: ${YELLOW}$PUSH${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Function to build backend
build_backend() {
    echo -e "${GREEN}Building Backend Service...${NC}"
    
    local DOCKERFILE="Dockerfile"
    local TARGET=""
    local REQUIREMENTS_FILE="requirements.txt"
    
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        DOCKERFILE="Dockerfile.dev"
        TARGET=""
        REQUIREMENTS_FILE="requirements.txt"
    else
        DOCKERFILE="Dockerfile"
        TARGET="production"
        REQUIREMENTS_FILE="requirements.render.txt"
    fi
    
    local IMAGE_NAME="$REGISTRY/$PROJECT_NAME-backend:$TAG"
    
    echo -e "${YELLOW}Image: $IMAGE_NAME${NC}"
    echo -e "${YELLOW}Dockerfile: backend/$DOCKERFILE${NC}"
    
    cd "$PROJECT_ROOT/backend"
    
    if [[ "$PLATFORM" == "both" ]]; then
        # Multi-platform build with buildx
        docker buildx build \
            $PLATFORM_FLAG \
            $NO_CACHE \
            --build-arg REQUIREMENTS_FILE=$REQUIREMENTS_FILE \
            ${TARGET:+--target $TARGET} \
            -t "$IMAGE_NAME" \
            -f "$DOCKERFILE" \
            ${PUSH:+--push} \
            .
    else
        # Single platform build
        docker build \
            $PLATFORM_FLAG \
            $NO_CACHE \
            --build-arg REQUIREMENTS_FILE=$REQUIREMENTS_FILE \
            ${TARGET:+--target $TARGET} \
            -t "$IMAGE_NAME" \
            -f "$DOCKERFILE" \
            .
        
        if [[ "$PUSH" == true ]]; then
            echo -e "${GREEN}Pushing backend image...${NC}"
            docker push "$IMAGE_NAME"
        fi
    fi
    
    echo -e "${GREEN}✓ Backend build complete${NC}\n"
}

# Function to build frontend
build_frontend() {
    echo -e "${GREEN}Building Frontend Service...${NC}"
    
    local IMAGE_NAME="$REGISTRY/$PROJECT_NAME-frontend:$TAG"
    local TARGET="production"
    
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        TARGET="development"
    fi
    
    echo -e "${YELLOW}Image: $IMAGE_NAME${NC}"
    echo -e "${YELLOW}Target: $TARGET${NC}"
    
    cd "$PROJECT_ROOT/frontend"
    
    # Load build args from .env if exists
    BUILD_ARGS=""
    if [[ -f ".env.production" ]]; then
        BUILD_ARGS="--build-arg VITE_API_URL=${VITE_API_URL:-http://localhost:8000} \
                    --build-arg VITE_WS_URL=${VITE_WS_URL:-ws://localhost:8000/ws}"
    fi
    
    if [[ "$PLATFORM" == "both" ]]; then
        # Multi-platform build with buildx
        docker buildx build \
            $PLATFORM_FLAG \
            $NO_CACHE \
            --target $TARGET \
            $BUILD_ARGS \
            -t "$IMAGE_NAME" \
            -f Dockerfile \
            ${PUSH:+--push} \
            .
    else
        # Single platform build
        docker build \
            $PLATFORM_FLAG \
            $NO_CACHE \
            --target $TARGET \
            $BUILD_ARGS \
            -t "$IMAGE_NAME" \
            -f Dockerfile \
            .
        
        if [[ "$PUSH" == true ]]; then
            echo -e "${GREEN}Pushing frontend image...${NC}"
            docker push "$IMAGE_NAME"
        fi
    fi
    
    echo -e "${GREEN}✓ Frontend build complete${NC}\n"
}

# Main build logic
cd "$PROJECT_ROOT"

if [[ "$SERVICE" == "backend" || "$SERVICE" == "all" ]]; then
    build_backend
fi

if [[ "$SERVICE" == "frontend" || "$SERVICE" == "all" ]]; then
    build_frontend
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}✓ Build Complete!${NC}"
echo -e "${BLUE}=====================================${NC}"

# Print next steps
echo -e "\n${YELLOW}Next steps:${NC}"
if [[ "$PUSH" == false ]]; then
    echo -e "  1. Test locally: ${BLUE}docker-compose up${NC}"
    echo -e "  2. Push to registry: ${BLUE}$0 --service $SERVICE --push${NC}"
else
    echo -e "  1. Pull on server: ${BLUE}docker pull $REGISTRY/$PROJECT_NAME-$SERVICE:$TAG${NC}"
    echo -e "  2. Deploy: ${BLUE}docker-compose up -d${NC}"
fi

exit 0
