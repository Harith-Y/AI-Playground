# GitHub Actions CI/CD Workflows

This directory contains all GitHub Actions workflows for continuous integration and deployment.

## Workflows Overview

### 1. Backend CI (`backend-ci.yml`)
**Triggers:** Push/PR to `main` or `develop` with backend changes

**Jobs:**
- **Lint**: Code quality checks with flake8, black, isort, mypy
- **Test**: Run pytest with coverage on Python 3.11 and 3.12
- **Security**: Safety and Bandit security scans
- **Build**: Build Docker image and upload as artifact

**Status Badge:**
```markdown
![Backend CI](https://github.com/Harith-Y/AI-Playground/workflows/Backend%20CI/badge.svg)
```

### 2. Frontend CI (`frontend-ci.yml`)
**Triggers:** Push/PR to `main` or `develop` with frontend changes

**Jobs:**
- **Lint**: ESLint, Prettier, TypeScript checks
- **Test**: Jest/React Testing Library on Node 18 and 20
- **Build**: Production build and Docker image
- **Lighthouse**: Performance audit (on PR)

**Status Badge:**
```markdown
![Frontend CI](https://github.com/Harith-Y/AI-Playground/workflows/Frontend%20CI/badge.svg)
```

### 3. Docker Compose CI (`docker-compose-ci.yml`)
**Triggers:** Push/PR to `main` or `develop` with docker changes

**Jobs:**
- **Validate**: Validate all docker-compose configurations
- **Integration Test**: Full stack integration testing
- **Security Scan**: Trivy vulnerability scanning

### 4. CD - Deploy (`cd-deploy.yml`)
**Triggers:**
- Push to `main` branch
- Git tags matching `v*.*.*`
- Manual workflow dispatch

**Jobs:**
- **Build and Push**: Build and push Docker images to GitHub Container Registry
- **Deploy Staging**: Deploy to staging environment (on main branch)
- **Deploy Production**: Deploy to production (on version tags)
- **Rollback**: Automatic rollback on deployment failure

**Environments:**
- `staging`: https://staging.ai-playground.example.com
- `production`: https://ai-playground.example.com

### 5. Code Quality (`code-quality.yml`)
**Triggers:**
- Push/PR to `main` or `develop`
- Weekly schedule (Sundays at 00:00 UTC)

**Jobs:**
- **CodeQL**: Security analysis for Python and JavaScript
- **SonarCloud**: Code quality and coverage analysis (optional)
- **Dependency Review**: Check for vulnerable dependencies
- **License Check**: Verify license compliance

### 6. PR Checks (`pr-checks.yml`)
**Triggers:** Pull request events

**Jobs:**
- **PR Title**: Validate semantic PR titles
- **PR Size**: Warn on large PRs
- **Auto-label**: Automatically label PRs based on changed files
- **Checklist**: Verify PR description completeness
- **Conflict Check**: Check for merge conflicts
- **TODO Check**: Flag TODO/FIXME comments
- **Breaking Changes**: Detect potential breaking changes

## Required Secrets

Configure these secrets in your GitHub repository settings:

### Deployment Secrets
```bash
# AWS (if using AWS deployment)
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

# SSH Deployment
STAGING_HOST
STAGING_USERNAME
STAGING_SSH_KEY

PRODUCTION_HOST
PRODUCTION_USERNAME
PRODUCTION_SSH_KEY

# Notifications
SLACK_WEBHOOK  # For deployment notifications
```

### Optional Secrets
```bash
# SonarCloud (for code quality analysis)
SONAR_TOKEN

# Container Registry (GitHub Container Registry is used by default)
REGISTRY_USERNAME
REGISTRY_PASSWORD
```

## Environment Variables

These are configured automatically or in the workflow files:

```yaml
REGISTRY: ghcr.io  # GitHub Container Registry
IMAGE_NAME: ${{ github.repository }}
```

## Workflow Badges

Add these badges to your README.md:

```markdown
[![Backend CI](https://github.com/Harith-Y/AI-Playground/workflows/Backend%20CI/badge.svg)](https://github.com/Harith-Y/AI-Playground/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/Harith-Y/AI-Playground/workflows/Frontend%20CI/badge.svg)](https://github.com/Harith-Y/AI-Playground/actions/workflows/frontend-ci.yml)
[![Docker Compose CI](https://github.com/Harith-Y/AI-Playground/workflows/Docker%20Compose%20CI/badge.svg)](https://github.com/Harith-Y/AI-Playground/actions/workflows/docker-compose-ci.yml)
[![Code Quality](https://github.com/Harith-Y/AI-Playground/workflows/Code%20Quality/badge.svg)](https://github.com/Harith-Y/AI-Playground/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/Harith-Y/AI-Playground/branch/main/graph/badge.svg)](https://codecov.io/gh/Harith-Y/AI-Playground)
```

## Manual Workflow Triggers

### Deploy to Staging
```bash
# Via GitHub CLI
gh workflow run cd-deploy.yml -f environment=staging

# Via GitHub UI
Actions > CD - Deploy > Run workflow > Select staging
```

### Deploy to Production
```bash
# Via GitHub CLI
gh workflow run cd-deploy.yml -f environment=production

# Or create a version tag
git tag v1.0.0
git push origin v1.0.0
```

## Local Testing

Test workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
choco install act  # Windows

# Test backend CI
act -W .github/workflows/backend-ci.yml

# Test with specific event
act pull_request -W .github/workflows/pr-checks.yml

# Test with secrets
act -s GITHUB_TOKEN=your_token
```

## Caching Strategy

Workflows use caching to speed up builds:

- **Python**: `pip` cache via `actions/setup-python@v5`
- **Node**: `npm` cache via `actions/setup-node@v4`
- **Docker**: Layer caching via `docker/build-push-action@v5` with GitHub Actions cache

## Artifacts

Workflows produce these artifacts:

1. **Backend Docker Image** (1 day retention)
2. **Frontend Docker Image** (1 day retention)
3. **Frontend Build** (7 days retention)
4. **Security Reports** (permanent)
5. **Test Coverage Reports** (uploaded to Codecov)
6. **License Reports** (permanent)

## Best Practices

1. **Branch Protection**: Enable required status checks for `main`:
   - Backend CI > test
   - Frontend CI > test
   - PR Checks > pr-title

2. **PR Reviews**: Require at least 1 approval before merging

3. **Semantic Versioning**: Use conventional commits and semantic versioning:
   - `feat:` → Minor version bump
   - `fix:` → Patch version bump
   - `BREAKING CHANGE:` → Major version bump

4. **Deployment Strategy**:
   - All PRs → Run CI checks
   - Merge to `main` → Deploy to staging
   - Create tag `v*.*.*` → Deploy to production

5. **Rollback Strategy**:
   - Keep previous Docker images tagged
   - Automatic rollback on deployment failure
   - Manual rollback via workflow dispatch

## Troubleshooting

### Common Issues

**1. Workflow not triggering**
- Check the `paths` filter matches your changes
- Verify branch name matches the workflow configuration

**2. Permission denied errors**
- Ensure repository has `Actions: Read and write permissions`
- Check if required secrets are configured

**3. Docker build failures**
- Verify Dockerfile syntax
- Check for missing dependencies in requirements.txt

**4. Test failures**
- Review test logs in the Actions tab
- Run tests locally to reproduce

**5. Deployment failures**
- Check server SSH connectivity
- Verify environment variables are set
- Review server logs

## Monitoring and Alerts

Configure notifications for workflow failures:

1. **Email**: GitHub sends email notifications by default
2. **Slack**: Configure webhook in repository secrets
3. **GitHub Mobile**: Install GitHub mobile app for push notifications

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Actions Marketplace](https://github.com/marketplace?type=actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
