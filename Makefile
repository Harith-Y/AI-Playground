# ====================================
# Makefile for AI-Playground Docker Builds
# ====================================
# Provides convenient commands for building, testing, and deploying Docker images
#
# Usage:
#   make build          - Build all services (production)
#   make build-dev      - Build all services (development)
#   make build-backend  - Build backend only
#   make build-frontend - Build frontend only
#   make push           - Build and push all images
#   make test           - Test Docker images
#   make clean          - Clean up Docker resources
#   make help           - Show this help

.PHONY: help build build-dev build-backend build-frontend push test clean health-check

# Variables
SHELL := /bin/bash
PROJECT_NAME := ai-playground
TAG ?= latest
ENV ?= prod
REGISTRY ?= docker.io
PLATFORM ?= linux/amd64

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "======================================"
	@echo "AI-Playground Docker Build Commands"
	@echo "======================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${BLUE}%-20s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  TAG=${YELLOW}${TAG}${NC}          - Image tag"
	@echo "  ENV=${YELLOW}${ENV}${NC}          - Environment (dev/prod)"
	@echo "  REGISTRY=${YELLOW}${REGISTRY}${NC} - Docker registry"
	@echo "  PLATFORM=${YELLOW}${PLATFORM}${NC} - Build platform"
	@echo ""
	@echo "Examples:"
	@echo "  ${GREEN}make build${NC}                    - Build all services"
	@echo "  ${GREEN}make build TAG=v1.0.0${NC}         - Build with custom tag"
	@echo "  ${GREEN}make build-dev${NC}                - Build development images"
	@echo "  ${GREEN}make push TAG=v1.0.0${NC}          - Build and push with tag"
	@echo ""

build: ## Build all services (production)
	@echo "${BLUE}Building all services...${NC}"
	@./docker/build/docker-build.sh --service all --env prod --tag $(TAG) --platform $(PLATFORM)

build-dev: ## Build all services (development)
	@echo "${BLUE}Building development images...${NC}"
	@./docker/build/docker-build.sh --service all --env dev --tag $(TAG)-dev --platform $(PLATFORM)

build-backend: ## Build backend service only
	@echo "${BLUE}Building backend service...${NC}"
	@./docker/build/docker-build.sh --service backend --env $(ENV) --tag $(TAG) --platform $(PLATFORM)

build-frontend: ## Build frontend service only
	@echo "${BLUE}Building frontend service...${NC}"
	@./docker/build/docker-build.sh --service frontend --env $(ENV) --tag $(TAG) --platform $(PLATFORM)

build-multi: ## Build multi-platform images (amd64 + arm64)
	@echo "${BLUE}Building multi-platform images...${NC}"
	@./docker/build/docker-build.sh --service all --env prod --platform both --tag $(TAG)

push: ## Build and push all images to registry
	@echo "${BLUE}Building and pushing all images...${NC}"
	@./docker/build/docker-build.sh --service all --env prod --push --tag $(TAG) --registry $(REGISTRY)

push-backend: ## Build and push backend image
	@echo "${BLUE}Building and pushing backend...${NC}"
	@./docker/build/docker-build.sh --service backend --env prod --push --tag $(TAG) --registry $(REGISTRY)

push-frontend: ## Build and push frontend image
	@echo "${BLUE}Building and pushing frontend...${NC}"
	@./docker/build/docker-build.sh --service frontend --env prod --push --tag $(TAG) --registry $(REGISTRY)

test: ## Test Docker images locally
	@echo "${BLUE}Testing Docker images...${NC}"
	@docker-compose up -d
	@sleep 10
	@echo "${YELLOW}Checking backend health...${NC}"
	@curl -f http://localhost:8000/health || (echo "${RED}Backend health check failed${NC}" && exit 1)
	@echo "${GREEN}✓ Backend healthy${NC}"
	@echo "${YELLOW}Checking frontend...${NC}"
	@curl -f http://localhost:80 || (echo "${RED}Frontend check failed${NC}" && exit 1)
	@echo "${GREEN}✓ Frontend accessible${NC}"
	@docker-compose down
	@echo "${GREEN}✓ All tests passed${NC}"

test-backend: ## Test backend image
	@echo "${BLUE}Testing backend image...${NC}"
	@docker run -d --name test-backend -p 8000:8000 $(PROJECT_NAME)-backend:$(TAG)
	@sleep 5
	@curl -f http://localhost:8000/health || (echo "${RED}Backend health check failed${NC}" && docker stop test-backend && docker rm test-backend && exit 1)
	@echo "${GREEN}✓ Backend test passed${NC}"
	@docker stop test-backend
	@docker rm test-backend

test-frontend: ## Test frontend image
	@echo "${BLUE}Testing frontend image...${NC}"
	@docker run -d --name test-frontend -p 80:80 $(PROJECT_NAME)-frontend:$(TAG)
	@sleep 5
	@curl -f http://localhost:80 || (echo "${RED}Frontend check failed${NC}" && docker stop test-frontend && docker rm test-frontend && exit 1)
	@echo "${GREEN}✓ Frontend test passed${NC}"
	@docker stop test-frontend
	@docker rm test-frontend

health-check: ## Run Docker build health check
	@echo "${BLUE}Running health check...${NC}"
	@if command -v powershell > /dev/null 2>&1; then \
		powershell -ExecutionPolicy Bypass -File ./docker/build/health-check.ps1; \
	else \
		echo "${RED}PowerShell not found. Please run health-check.ps1 manually.${NC}"; \
	fi

clean: ## Clean up Docker resources (images, containers, volumes)
	@echo "${YELLOW}Cleaning up Docker resources...${NC}"
	@docker-compose down -v 2>/dev/null || true
	@docker system prune -f
	@echo "${GREEN}✓ Cleanup complete${NC}"

clean-images: ## Remove project Docker images
	@echo "${YELLOW}Removing project images...${NC}"
	@docker images | grep $(PROJECT_NAME) | awk '{print $$3}' | xargs -r docker rmi -f || true
	@echo "${GREEN}✓ Images removed${NC}"

clean-all: clean clean-images ## Full cleanup (images, containers, volumes, networks)
	@echo "${YELLOW}Performing full cleanup...${NC}"
	@docker system prune -a -f --volumes
	@echo "${GREEN}✓ Full cleanup complete${NC}"

up: ## Start all services with docker-compose
	@echo "${BLUE}Starting services...${NC}"
	@docker-compose up -d
	@echo "${GREEN}✓ Services started${NC}"
	@docker-compose ps

down: ## Stop all services
	@echo "${YELLOW}Stopping services...${NC}"
	@docker-compose down
	@echo "${GREEN}✓ Services stopped${NC}"

logs: ## Show logs from all services
	@docker-compose logs -f

logs-backend: ## Show backend logs
	@docker-compose logs -f backend

logs-frontend: ## Show frontend logs
	@docker-compose logs -f frontend

ps: ## Show running containers
	@docker-compose ps

rebuild: clean-images build ## Clean and rebuild all images
	@echo "${GREEN}✓ Rebuild complete${NC}"

validate: ## Validate Docker configurations
	@echo "${BLUE}Validating Docker configurations...${NC}"
	@docker-compose config > /dev/null && echo "${GREEN}✓ docker-compose.yml is valid${NC}" || echo "${RED}✗ docker-compose.yml is invalid${NC}"
	@docker-compose -f docker-compose.dev.yml config > /dev/null && echo "${GREEN}✓ docker-compose.dev.yml is valid${NC}" || echo "${RED}✗ docker-compose.dev.yml is invalid${NC}"
	@docker-compose -f docker-compose.production.yml config > /dev/null && echo "${GREEN}✓ docker-compose.production.yml is valid${NC}" || echo "${RED}✗ docker-compose.production.yml is invalid${NC}"

size: ## Show Docker image sizes
	@echo "${BLUE}Docker Image Sizes:${NC}"
	@docker images | grep $(PROJECT_NAME) | awk '{printf "  %-40s %s\n", $$1":"$$2, $$7" "$$8}'

version: ## Show Docker and tool versions
	@echo "${BLUE}Versions:${NC}"
	@echo "  Docker: $$(docker --version)"
	@echo "  Docker Compose: $$(docker-compose --version)"
	@if command -v docker buildx version > /dev/null 2>&1; then \
		echo "  Docker Buildx: $$(docker buildx version)"; \
	else \
		echo "  Docker Buildx: ${RED}Not installed${NC}"; \
	fi

.DEFAULT_GOAL := help
