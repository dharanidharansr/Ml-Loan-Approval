from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Clean and convert form values - remove commas and convert to float
        features = []
        for x in request.form.values():
            # Remove commas and any other non-numeric characters except decimal point
            cleaned_value = str(x).replace(',', '').replace(' ', '')
            features.append(float(cleaned_value))
        
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        
        result = 'Approved' if prediction[0] == 0 else 'Not Approved'
        return render_template('index.html', prediction=result)
        
    except ValueError as e:
        # Handle conversion errors gracefully
        error_message = f"Please enter valid numbers in all fields. Error: {str(e)}"
        return render_template('index.html', error=error_message)
    except Exception as e:
        # Handle any other errors
        error_message = f"An error occurred during prediction. Please try again."
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)