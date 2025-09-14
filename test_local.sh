#!/bin/bash

# Local development and testing script
# This script helps you run tests and checks locally

set -e

echo "🧪 Local Development Test Script"
echo "================================"

# Function to print colored output
print_status() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

# Check if we're in the right directory
if [[ ! -f "app.py" ]]; then
    print_error "app.py not found. Please run this script from the project root directory."
    exit 1
fi

print_status "Checking Python syntax..."

# Basic Python syntax check
python3 -m py_compile app.py
if [[ $? -eq 0 ]]; then
    print_status "✅ app.py syntax is valid"
else
    print_error "❌ app.py has syntax errors"
    exit 1
fi

# Check if required files exist
print_status "Checking required files..."
required_files=("app.py" "requirements.txt" "model.pkl" "Procfile" "templates/index.html")

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_status "✅ $file exists"
    else
        print_warning "⚠️  $file is missing"
    fi
done

# Check imports
print_status "Checking Python imports..."
python3 -c "
try:
    import flask
    print('✅ Flask is available')
except ImportError:
    print('❌ Flask is not installed')

try:
    import numpy
    print('✅ NumPy is available')
except ImportError:
    print('❌ NumPy is not installed')

try:
    import pickle
    print('✅ Pickle is available')
except ImportError:
    print('❌ Pickle is not installed')
"

# Test model loading
print_status "Testing model loading..."
python3 -c "
try:
    import pickle
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('✅ Model loads successfully')
    print(f'✅ Model type: {type(model).__name__}')
    if hasattr(model, 'predict'):
        print('✅ Model has predict method')
    else:
        print('⚠️  Model missing predict method')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"

# Test Flask app imports
print_status "Testing Flask app..."
python3 -c "
try:
    from app import app
    print('✅ Flask app imports successfully')
    
    # Test routes
    with app.test_client() as client:
        response = client.get('/')
        if response.status_code == 200:
            print('✅ Home route works')
        else:
            print(f'❌ Home route failed: {response.status_code}')
            
        response = client.get('/health')
        if response.status_code == 200:
            print('✅ Health route works')
        else:
            print(f'❌ Health route failed: {response.status_code}')
            
except Exception as e:
    print(f'❌ Flask app test failed: {e}')
"

print_status "🎉 Local tests completed!"
print_status "📋 Summary:"
print_status "   - If all checks passed, your app should work in CI/CD"
print_status "   - If any checks failed, fix them before pushing to GitHub"
print_status "   - The GitHub Actions workflow will run similar checks automatically"
