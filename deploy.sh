#!/bin/bash

# Deployment helper script for Loan Approval App
# Usage: ./deploy.sh [staging|production]

set -e

ENVIRONMENT=${1:-production}
PROJECT_NAME="loan-approval-app"

echo "🚀 Starting deployment to $ENVIRONMENT..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if git is clean
if [[ -n $(git status --porcelain) ]]; then
    print_warning "Working directory is not clean. Uncommitted changes detected."
    echo "Uncommitted files:"
    git status --porcelain
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests before deployment
print_status "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v
    if [ $? -ne 0 ]; then
        print_error "Tests failed! Aborting deployment."
        exit 1
    fi
    print_status "All tests passed ✅"
else
    print_warning "pytest not found. Skipping tests."
fi

# Check code quality
print_status "Running code quality checks..."
if command -v flake8 &> /dev/null; then
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    print_status "Code quality check passed ✅"
else
    print_warning "flake8 not found. Skipping code quality check."
fi

# Check if required files exist
required_files=("app.py" "requirements.txt" "model.pkl" "Procfile")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "Required file $file not found!"
        exit 1
    fi
done
print_status "All required files present ✅"

# Create deployment tag
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TAG="deploy-${ENVIRONMENT}-${TIMESTAMP}"

print_status "Creating deployment tag: $TAG"
git tag -a "$TAG" -m "Deployment to $ENVIRONMENT on $(date)"

# Push to GitHub (triggers CI/CD)
print_status "Pushing to GitHub..."
git push origin main
git push origin "$TAG"

print_status "🎉 Deployment initiated successfully!"
print_status "Check GitHub Actions for deployment progress:"
print_status "https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"

# Wait for deployment hook (if URL is provided)
if [[ -n "$RENDER_DEPLOY_HOOK_URL" ]]; then
    print_status "Triggering Render deployment..."
    curl -X POST "$RENDER_DEPLOY_HOOK_URL"
    print_status "Render deployment triggered ✅"
else
    print_warning "RENDER_DEPLOY_HOOK_URL not set. Manual trigger or GitHub Actions will handle deployment."
fi

# Final instructions
echo
print_status "Deployment Summary:"
echo "- Environment: $ENVIRONMENT"
echo "- Tag: $TAG"
echo "- Timestamp: $(date)"
echo
print_status "Next steps:"
echo "1. Monitor GitHub Actions workflow"
echo "2. Check Render deployment logs"
echo "3. Verify application health after deployment"
echo "4. Run post-deployment tests if needed"
