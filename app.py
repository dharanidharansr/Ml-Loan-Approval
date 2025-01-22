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
    
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)

    
    result = 'Approved' if prediction[0] == 0 else 'Not Approved'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)