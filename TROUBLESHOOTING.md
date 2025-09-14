# 🔧 CI/CD Troubleshooting Guide

## Common Issues and Solutions

### 1. Flake8 Configuration Errors

**Problem**: `ValueError: Error code '#' supplied to 'ignore' option does not match '^[A-Z]{1,3}[0-9]{0,3}$'`

**Cause**: Inline comments in `.flake8` configuration file are not supported.

**Solution**: 
- Remove inline comments from configuration values
- Use separate comment lines or no comments

**❌ Wrong:**
```ini
ignore = 
    E203,  # whitespace before ':'
    W503,  # line break before binary operator
```

**✅ Correct:**
```ini
ignore = 
    E203,
    W503,
```

### 2. Virtual Environment Linting Issues

**Problem**: Flake8 checking virtual environment files and reporting false positives.

**Cause**: Linting entire project directory including `venv/`, `ml/`, or other virtual environments.

**Solution**:
- Exclude virtual environment directories in `.flake8`
- Use specific file targeting in GitHub Actions
- Add virtual environment to `.gitignore`

**Example exclude patterns:**
```ini
[flake8]
exclude = 
    ml,
    venv,
    .venv,
    env,
    __pycache__,
    .git
```

### 3. Import Errors in Tests

**Problem**: `ImportError: No module named 'sklearn'` or similar in CI.

**Cause**: Dependencies not properly installed or virtual environment issues.

**Solutions**:
- Ensure all dependencies are in `requirements.txt`
- Make imports optional with try/except in tests
- Use `pytest.skip()` for missing optional dependencies

**Example:**
```python
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@pytest.fixture
def mock_model():
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not available")
    # ... rest of fixture
```

### 4. GitHub Secrets Missing

**Problem**: `curl: (7) Failed to connect` or deployment failures.

**Cause**: Required secrets not configured in GitHub repository.

**Solution**:
1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add required secrets:
   - `RENDER_DEPLOY_HOOK_URL`
   - `APP_URL`
   - Optional: `SLACK_WEBHOOK`, `TEAMS_WEBHOOK`, `SNYK_TOKEN`

### 5. Model Loading Issues

**Problem**: Tests fail because `model.pkl` cannot be loaded.

**Cause**: Model file missing, corrupted, or incompatible versions.

**Solutions**:
- Ensure `model.pkl` is committed to repository
- Check file permissions and size
- Verify model was saved with compatible scikit-learn version
- Add model validation in tests

### 6. File Path Issues in CI

**Problem**: Tests work locally but fail in CI with file not found errors.

**Cause**: Different working directories or path separators.

**Solution**:
- Use `os.path.join()` for cross-platform paths
- Use absolute paths or relative to script location
- Check working directory in tests

**Example:**
```python
import os
import sys

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
```

### 7. Workflow Permissions Issues

**Problem**: GitHub Actions fails with permission errors.

**Cause**: Repository or workflow permissions not properly configured.

**Solution**:
- Check repository settings → Actions → General
- Ensure workflows have read/write permissions
- Verify branch protection rules don't block automation

## Debugging Tips

### 1. Local Testing
Run the local test script before pushing:
```bash
./test_local.sh
```

### 2. GitHub Actions Debugging
- Check the "Actions" tab for detailed logs
- Look at each job separately (test, security-scan, build-and-deploy)
- Use `echo` statements in workflows for debugging

### 3. Step-by-step Testing
Test components individually:
```bash
# Syntax check
python3 -c "import py_compile; py_compile.compile('app.py', doraise=True)"

# Import check
python3 -c "from app import app; print('✅ App imports successfully')"

# Model check
python3 -c "import pickle; model = pickle.load(open('model.pkl', 'rb')); print('✅ Model loads')"
```

### 4. Flake8 Testing
Test flake8 configuration:
```bash
# Test on specific files
flake8 app.py --select=E9,F63,F7,F82

# Test configuration
flake8 --help | grep config
```

## Prevention Strategies

### 1. Pre-commit Hooks
Install pre-commit hooks to catch issues early:
```bash
pip install pre-commit
pre-commit install
```

### 2. Local Environment Setup
Keep local environment similar to CI:
```bash
# Use same Python version as CI
python3.11 -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
```

### 3. Regular Testing
- Test locally before every commit
- Use feature branches for major changes
- Monitor CI/CD pipeline regularly

### 4. Documentation
- Keep this troubleshooting guide updated
- Document any project-specific issues
- Share knowledge with team members

## Quick Fixes Checklist

When CI fails, check these in order:

1. **☐ Syntax errors**: Run `python3 -c "import py_compile; py_compile.compile('app.py', doraise=True)"`
2. **☐ File existence**: Verify all required files are committed
3. **☐ Dependencies**: Check `requirements.txt` is up to date
4. **☐ Configuration**: Validate `.flake8`, `pytest.ini` syntax
5. **☐ Secrets**: Ensure GitHub secrets are configured
6. **☐ Paths**: Check file paths are correct and cross-platform
7. **☐ Permissions**: Verify GitHub Actions permissions
8. **☐ Logs**: Read full error logs in GitHub Actions

Following this guide should help resolve most CI/CD issues quickly!
