# CI/CD Pipeline Setup Guide

This document explains how to set up and configure the CI/CD pipeline for your Loan Approval Prediction application.

## Overview

The CI/CD pipeline is built using GitHub Actions and includes:
- Automated testing
- Code quality checks
- Security scanning
- Automated deployment to Render
- Notifications

## Pipeline Features

### 1. Continuous Integration (CI)
- **Testing**: Automated unit tests using pytest
- **Code Quality**: Linting with flake8
- **Coverage**: Code coverage reporting
- **Security**: Vulnerability scanning with Snyk

### 2. Continuous Deployment (CD)
- **Automatic Deployment**: Deploys to Render on main branch pushes
- **Health Checks**: Validates deployment success
- **Notifications**: Slack/Teams notifications

## Setup Instructions

### 1. GitHub Repository Setup

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit with CI/CD pipeline"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

### 2. Render Setup

1. **Get Deploy Hook URL**:
   - Go to your Render dashboard
   - Navigate to your service
   - Go to Settings > Build & Deploy
   - Copy the "Deploy Hook" URL

2. **Set up Health Check Endpoint**:
   - Note your application URL from Render

### 3. GitHub Secrets Configuration

Add the following secrets to your GitHub repository (Settings > Secrets and variables > Actions):

#### Required Secrets:
```
RENDER_DEPLOY_HOOK_URL=https://api.render.com/deploy/srv-xxxxx?key=xxxxx
APP_URL=https://your-app-name.onrender.com
```

#### Optional Secrets (for enhanced features):
```
RENDER_STAGING_DEPLOY_HOOK_URL=https://api.render.com/deploy/srv-staging-xxxxx?key=xxxxx
STAGING_APP_URL=https://your-staging-app.onrender.com
SLACK_WEBHOOK=https://hooks.slack.com/services/xxx/xxx/xxx
TEAMS_WEBHOOK=https://your-teams-webhook-url
SNYK_TOKEN=your-snyk-token-for-security-scanning
```

### 4. Workflow Triggers

The pipeline triggers on:
- **Push to main branch**: Full CI/CD pipeline
- **Push to develop branch**: CI only
- **Pull requests to main**: CI only
- **Manual deployment**: Use the deploy.yml workflow

## Workflow Files

### 1. `.github/workflows/ci-cd.yml`
Main CI/CD pipeline that:
- Runs tests on Python 3.11
- Performs code quality checks
- Runs security scans
- Deploys to production (main branch only)
- Sends notifications

### 2. `.github/workflows/deploy.yml`
Manual deployment workflow for:
- Staging deployments
- Production deployments
- Environment-specific configurations

## Local Development

### 1. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install pre-commit
```

### 2. Set up Pre-commit Hooks
```bash
pre-commit install
```

### 3. Run Tests Locally
```bash
pytest
```

### 4. Run Linting
```bash
flake8 .
```

## Testing

The test suite includes:
- **Unit Tests**: Testing Flask routes and functionality
- **Integration Tests**: Testing model loading and predictions
- **Security Tests**: Basic XSS and security checks

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_app.py
```

## Security

The pipeline includes:
- **Snyk Security Scanning**: Vulnerability detection
- **Code Quality Checks**: Static analysis
- **Dependency Scanning**: Known vulnerability checks

## Monitoring and Notifications

### Slack Integration
1. Create a Slack app and get webhook URL
2. Add `SLACK_WEBHOOK` secret to GitHub
3. Notifications will be sent on deployment success/failure

### Teams Integration
1. Create Teams incoming webhook
2. Add `TEAMS_WEBHOOK` secret to GitHub
3. Configure in deploy.yml workflow

## Environment Management

The pipeline supports multiple environments:
- **Development**: Local development
- **Staging**: Testing environment (optional)
- **Production**: Live application on Render

## Troubleshooting

### Common Issues:

1. **Tests Failing**:
   - Check model.pkl exists
   - Verify test data format matches model expectations
   - Check dependencies in requirements.txt

2. **Deployment Failures**:
   - Verify RENDER_DEPLOY_HOOK_URL is correct
   - Check Render service logs
   - Ensure all required files are included

3. **Security Scan Issues**:
   - Review Snyk scan results
   - Update dependencies with vulnerabilities
   - Add SNYK_TOKEN secret for detailed scanning

### Debug Commands:
```bash
# Check workflow status
gh workflow list

# View workflow runs
gh run list

# View specific run details
gh run view <run-id>
```

## Best Practices

1. **Branch Protection**: Set up branch protection rules for main branch
2. **Required Checks**: Make CI checks required for merge
3. **Code Reviews**: Require pull request reviews
4. **Environment Secrets**: Use different secrets for staging/production
5. **Monitoring**: Set up application monitoring on Render

## Customization

### Adding New Tests
1. Create test files in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures and assertions

### Adding New Deployment Targets
1. Update workflows with new environment
2. Add corresponding secrets
3. Configure health checks

### Modifying Notification Channels
1. Update workflow files
2. Add webhook URLs to secrets
3. Customize message format

## Support

For issues or questions:
1. Check GitHub Actions workflow logs
2. Review Render deployment logs
3. Consult GitHub Actions documentation
4. Check Render deployment documentation
