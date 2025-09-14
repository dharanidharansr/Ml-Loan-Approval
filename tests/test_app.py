import pytest
import sys
import os
import tempfile
import pickle
import numpy as np

# Add the parent directory to sys.path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import sklearn, but make it optional for basic tests
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RandomForestClassifier = None

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not available")
    
    # Create a simple mock model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Create dummy training data
    X = np.random.rand(100, 10)  # Assuming 10 features
    y = np.random.randint(0, 2, 100)  # Binary classification
    model.fit(X, y)
    
    # Save the mock model temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        pickle.dump(model, f)
        return f.name


class TestFlaskApp:
    """Test cases for the Flask application."""
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data
        assert 'timestamp' in data
    
    def test_home_route(self, client):
        """Test the home route returns the correct template."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'<!DOCTYPE html>' in response.data or b'<html>' in response.data
    
    def test_predict_route_get_method(self, client):
        """Test that GET request to predict route is not allowed."""
        response = client.get('/predict')
        assert response.status_code == 405  # Method Not Allowed
    
    def test_predict_route_valid_data(self, client):
        """Test prediction with valid numerical data."""
        # Sample data for prediction (adjust based on your model's expected features)
        test_data = {
            'feature1': '1000',
            'feature2': '2000',
            'feature3': '50000',
            'feature4': '25',
            'feature5': '1',
            'feature6': '0',
            'feature7': '1',
            'feature8': '100',
            'feature9': '500',
            'feature10': '1'
        }
        
        response = client.post('/predict', data=test_data)
        assert response.status_code == 200
        # Check if response contains either 'Approved' or 'Not Approved'
        response_text = response.data.decode('utf-8')
        assert 'Approved' in response_text or 'Not Approved' in response_text
    
    def test_predict_route_with_commas(self, client):
        """Test prediction with comma-separated numbers."""
        test_data = {
            'feature1': '1,000',
            'feature2': '2,000',
            'feature3': '50,000',
            'feature4': '25',
            'feature5': '1',
            'feature6': '0',
            'feature7': '1',
            'feature8': '100',
            'feature9': '500',
            'feature10': '1'
        }
        
        response = client.post('/predict', data=test_data)
        assert response.status_code == 200
        response_text = response.data.decode('utf-8')
        assert 'Approved' in response_text or 'Not Approved' in response_text
    
    def test_predict_route_invalid_data(self, client):
        """Test prediction with invalid data."""
        test_data = {
            'feature1': 'invalid',
            'feature2': 'data',
            'feature3': 'test',
            'feature4': '25',
            'feature5': '1',
            'feature6': '0',
            'feature7': '1',
            'feature8': '100',
            'feature9': '500',
            'feature10': '1'
        }
        
        response = client.post('/predict', data=test_data)
        assert response.status_code == 200
        response_text = response.data.decode('utf-8')
        assert 'error' in response_text.lower() or 'Error' in response_text
    
    def test_predict_route_empty_data(self, client):
        """Test prediction with empty data."""
        response = client.post('/predict', data={})
        assert response.status_code == 200
        # Should handle gracefully and show error
    
    def test_predict_route_missing_features(self, client):
        """Test prediction with missing features."""
        test_data = {
            'feature1': '1000',
            'feature2': '2000',
            # Missing other features
        }
        
        response = client.post('/predict', data=test_data)
        assert response.status_code == 200
        # Should handle gracefully


class TestModelIntegration:
    """Test cases for model integration."""
    
    def test_model_exists(self):
        """Test that the model file exists."""
        model_path = 'model.pkl'
        assert os.path.exists(model_path), "Model file 'model.pkl' should exist"
    
    def test_model_loads_successfully(self):
        """Test that the model can be loaded successfully."""
        try:
            with open('model.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            assert model is not None
            assert hasattr(model, 'predict'), "Model should have a predict method"
        except Exception as e:
            pytest.fail(f"Model loading failed: {str(e)}")
    
    def test_model_prediction_format(self):
        """Test that model returns predictions in expected format."""
        try:
            with open('model.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            
            # Create dummy input data (adjust dimensions based on your model)
            dummy_input = np.array([[1000, 2000, 50000, 25, 1, 0, 1, 100, 500, 1]])
            prediction = model.predict(dummy_input)
            
            assert prediction is not None
            assert len(prediction) > 0
            assert isinstance(prediction[0], (int, np.integer, float, np.floating))
        except Exception as e:
            pytest.fail(f"Model prediction test failed: {str(e)}")


class TestApplicationSecurity:
    """Test cases for application security."""
    
    def test_no_debug_mode(self):
        """Test that debug mode is disabled in production."""
        assert not app.config.get('DEBUG', False)
    
    def test_xss_protection(self, client):
        """Test basic XSS protection."""
        malicious_data = {
            'feature1': '<script>alert("xss")</script>',
            'feature2': '2000',
            'feature3': '50000',
            'feature4': '25',
            'feature5': '1',
            'feature6': '0',
            'feature7': '1',
            'feature8': '100',
            'feature9': '500',
            'feature10': '1'
        }
        
        response = client.post('/predict', data=malicious_data)
        assert response.status_code == 200
        response_text = response.data.decode('utf-8')
        # Should not execute script tags
        assert '<script>' not in response_text
